"""
    Train the NN-basedmask generator.
"""

import argparse
import glob
import importlib.util
import logging
import math
import os
import random
from pathlib import Path
from pdb import set_trace

import coloredlogs
import enlighten
import numpy as np
import seaborn as sns
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from sklearn.mixture import GaussianMixture
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import io
from tqdm import tqdm

from dnn.dnn_factory import DNN_Factory

# from dnn.fasterrcnn_resnet50 import FasterRCNN_ResNet50_FPN
# from dnn.fcn_resnet50 import FCN_ResNet50
from utilities.bbox_utils import center_size
from utilities.dataset import *
from utilities.loss_utils import get_mean_std
from utilities.loss_utils import shifted_mse as get_loss
from utilities.mask_utils import *
from utilities.results_utils import read_results
from utilities.timer import Timer
from utilities.video_utils import get_qp_from_name, read_videos, write_video
from utilities.visualize_utils import *

sns.set()

weight = [1, 1]

logger = logging.getLogger("train_detect_360p")
# thresh_list = torch.tensor([5, 7.5, 10])

dataset_direc = '/data02/wangweijun/RegenHance/'   ## please replace your path

def get_groundtruths(args, train_val_set, path, visualize_step_size, tag):
    app = DNN_Factory().get_model(args.app)
    loader = torch.utils.data.DataLoader(
        train_val_set,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=my_collate,
    )
    progress_bar = enlighten.get_manager().counter(
        total=len(train_val_set),
        desc=f"Generating saliency as ground truths",
        unit="frames",
    )
    saliency = {}

    # saliency = {}
    # for ground_truth in glob.glob(args.ground_truth + "*"):
    #     with open(ground_truth, "rb") as f:
    #         saliency.update(pickle.load(f))

    for data in loader:
        progress_bar.update()
        # get data
        if data == None:
            continue
        fid = data["fid"].item()
        real_id = data["real_id"].item()
        '''
        if args.local_rank >= 0 and fid % 2 != args.local_rank:
            continue
        '''
        vname = data["video_name"][0]

        sr_image = data["sr"].cuda(non_blocking=True)
        in_image = data["in"].cuda(non_blocking=True)
        sr_image.requires_grad = True

        with Timer("gt", logger):

            #with torch.no_grad():
            #    sr_result = app.inference(sr_image, detach=False, grad=True)
            with torch.enable_grad():
                if hasattr(torch.cuda, 'empty_cache'):    
                    torch.cuda.empty_cache()
                sr_result = app.inference(sr_image, detach=False, grad=True)
                sr_result = app.filter_result(
                    sr_result, args, class_check=args.class_check
                )

                if len(sr_result["instances"]) == 0:
                    (sr_image * 0.0).sum().backward()
                else:
                    sum(sr_result["instances"].scores).backward()

        sr_result["instances"] = sr_result["instances"].to("cpu")
        for key in sr_result["instances"].get_fields():
            if key == "pred_boxes":
                sr_result["instances"].get_fields()[key].tensor = (
                    sr_result["instances"].get_fields()[key].tensor.detach()
                )
            else:
                try:
                    sr_result["instances"].get_fields()[key] = (
                        sr_result["instances"].get_fields()[key].detach()
                    )
                except AttributeError:
                    print(key)
                    print(sr_result["instances"].get_fields()[key])
        
        mask_grad_ = sr_image.grad.norm(dim=1, p=1, keepdim=True)
                
        mask_grad = F.conv2d(
            mask_grad_,
            torch.ones([1, 1, args.tile_size, args.tile_size]).cuda(),
            stride=args.tile_size
        )
        
        mask_grad = mask_grad.detach().cpu()
        
        scores2grads = []
        regions = center_size(sr_result["instances"].pred_boxes.tensor)

        for i in range(regions.shape[0]):

            region = regions[i : i + 1, :]

            region_mask = generate_mask_from_regions(
                mask_grad.clone(), region, 0, args.tile_size, cuda=False
            )
            heat = mask_grad[region_mask > 0.5].mean()
            scores2grads.append(
                (sr_result["instances"].scores[i].item(), heat.item())
            )

        saliency[(vname, fid)] = {
            "saliency": mask_grad,
            "sr_result": sr_result,
            # "mean": mean,
            # "std": std,
            "scores2grads": scores2grads,
        }

        # visualize the saliency
        if fid % visualize_step_size == 0:

            # visualize
            if False:
            #if args.visualize:
                image = T.ToPILImage()(data["sr"][0])
                image_srresult = app.visualize(
                    image,
                    app.filter_result(
                        sr_result, args, class_check=args.class_check
                    ),
                )

                # plot the ground truth
                visualize_heat(
                    image_srresult,
                    mask_grad,
                    f"{path}/{fid}_saliency.jpg",
                    args,
                )

                visualize_heat(
                    image_srresult,
                    (mask_grad > 3).float(),
                    f"{path}/{fid}_gt.jpg",
                    args,
                )
                
                visualize_heat(
                    mask_grad.log(),
                    f"{path}/{fid}_log_saliency.jpg",
                    args,
                )

                visualize_dist(
                    mask_grad, f"{path}/{fid}_dist.jpg",
                )

                visualize_scores2grads(
                    scores2grads, f"{path}/{fid}_scores2grads.jpg",
                )

                visualize_log_dist(
                    mask_grad, f"{path}/{fid}_logdist.jpg",
                )

    # write saliency to disk
    with open(args.ground_truth + f".{tag}{args.local_rank}", "wb") as f:
        pickle.dump(saliency, f)
    #torch.save(VIS_GT, "VIS_GT_part2.pth")

def unzip_data(data, saliency):

    if data is None:
        raise ValueError

    fids = [fid.item() for fid in data["fid"]]
    names = [name for name in data["video_name"]]

    target = torch.cat(
        [saliency[(vname, fid)]["saliency"] for vname, fid in zip(names, fids)]
    )
    
    thresh_list = []

    sr_image = data["sr"]
    o_image = data["in"]

    return fids, names, sr_image, o_image, target, thresh_list


def visualize_test(fid, sr_image, mask_slice):
    maxid = 0
    image = T.ToPILImage()(sr_image[maxid])
    mask_slice = mask_slice[maxid : maxid + 1, :, :, :]
    mask_slice = mask_slice.softmax(dim=1)[:, 1:2, :, :]

    visualize_heat(
        image,
        mask_slice.cpu().detach(),
        f"train/{args.path}/test/{fid}_test.jpg",
        args,
    )


def visualize(maxid, fids, sr_image, mask_slice, target, saliency, tag):
    fid = fids[maxid]
    image = T.ToPILImage()(sr_image[maxid])
    mask_slice = mask_slice[maxid : maxid + 1, :, :, :]
    mask_slice = mask_slice.softmax(dim=1)[:, 1:2, :, :]
    target = target[maxid : maxid + 1, :, :, :]
    
    target = (target > 3).float()

    visualize_heat(
        image,
        mask_slice.cpu().detach(),
        f"train/{args.path}/{tag}/{fid}_train.jpg",
        args,
    )

    visualize_heat(
        image,
        target.cpu().detach(),
        f"train/{args.path}/{tag}/{fid}_saliency.jpg",
        args,
    )


def main(args):

    # initialize logger
    logger.addHandler(logging.FileHandler(args.log))
    torch.set_default_tensor_type(torch.FloatTensor)
    train_writer = SummaryWriter("runs/train")
    cross_writer = SummaryWriter("runs/cross")
    test_writer = SummaryWriter("runs/test")

    if args.training_set == "COCO":

        train_val_set = COCO_Dataset()
        # downsample original dataset
        train_val_set, _ = torch.utils.data.random_split(
            train_val_set,
            [
                math.ceil(0.2 * len(train_val_set)),
                math.floor(0.8 * len(train_val_set)),
            ],
            generator=torch.Generator().manual_seed(100),
        )

        logger.info("Dataset size: %d", len(train_val_set))

        training_set, cross_validation_set = torch.utils.data.random_split(
            train_val_set,
            [
                math.ceil(0.7 * len(train_val_set)),
                math.floor(0.3 * len(train_val_set)),
            ],
            generator=torch.Generator().manual_seed(100),
        )

    elif args.training_set == "CityScape":

        training_set = CityScape(train=True)
        cross_validation_set = CityScape(train=False)
        train_val_set = ConcatDataset([training_set, cross_validation_set])
    else:
        train_val_set = Test(root=dataset_direc)
        logger.info("Dataset size: %d", len(train_val_set))
        training_set, cross_validation_set = torch.utils.data.random_split(
            train_val_set,
            [
                math.ceil(0.7 * len(train_val_set)),
                math.floor(0.3 * len(train_val_set)),
            ],
            generator=torch.Generator().manual_seed(100),
        )


    test_set = get_testset(args.test_set)
    test_set, _ = torch.utils.data.random_split(
        test_set,
        [math.ceil(0.01 * len(test_set)), math.floor(0.99 * len(test_set)),],
        generator=torch.Generator().manual_seed(100),
    )

    training_loader = DataLoader(
        training_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=my_collate,
        pin_memory=True,
    )
    cross_validation_loader = DataLoader(
        cross_validation_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=my_collate,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=1,
        num_workers=args.num_workers,
        collate_fn=my_collate,
        pin_memory=True,
    )

    # construct the mask generator
    maskgen_spec = importlib.util.spec_from_file_location(
        "maskgen", args.maskgen_file
    )
    maskgen = importlib.util.module_from_spec(maskgen_spec)
    maskgen_spec.loader.exec_module(maskgen)
    mask_generator = maskgen.FCN(args.architecture)
    if args.init != "" and os.path.exists(args.init):
        logger.info(f"Load the model from %s", args.init)
        mask_generator.load(args.init)
    mask_generator.train()
    # mask_generator = nn.DataParallel(mask_generator)

    # mask_generator = torch.nn.parallel.DistributedDataParallel(mask_generator, device_ids=[args.local_rank])

    optimizer = torch.optim.Adam(
        mask_generator.parameters(), lr=args.learning_rate
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")

    # load ground truth results
    saliency = {}

    if len(glob.glob(args.ground_truth + "*")) != 0:
        saliency = {}
        for ground_truth in glob.glob(args.ground_truth + "*"):
            with open(ground_truth, "rb") as f:
                saliency.update(pickle.load(f))
    else:
        # get the application
        # generate saliency
        if args.local_rank <= 0:
            # only the master thread needs to calculate the groundtruth for test set.
            get_groundtruths(
                args, test_set, f"train/{args.path}/test/", 5, "testvideo"
            )
        get_groundtruths(
            args, train_val_set, f"train/{args.path}/VTest/", 10000, "VTest"
        )
        # return

    # training
    torch.cuda.empty_cache()

    mask_generator.cuda()
    mean_cross_validation_loss_before = 100

    overfitting_counter = 0

    for iteration in range(args.num_iterations):

        """
            Training
        """

        progress_bar = tqdm(
            total=len(training_set),
            desc=f"Iteration {iteration} on training set",
        )
        training_losses = []

        mask_generator.train()

        for idx, data in enumerate(training_loader):
            # break

            progress_bar.update(args.batch_size)

            try:
                fids, names, sr_image, o_image, target, _ = unzip_data(data, saliency)
            except ValueError:
                continue
            with torch.enable_grad():
                temp = o_image.cuda()
                if temp.shape[0] != 4:
                    logger.info(f"Wrong shape: {temp.shape}")
                    continue
                
                with Timer("train", logger):

                    mask_slice = mask_generator(temp)

                    # calculate loss
                    # loss = get_loss(mask_slice, target.cuda(), thresh_list.cuda())
                    loss = get_loss(mask_slice, target.cuda())
                    #print(loss.item())
                    loss.backward()

            # optimization and logging
            if idx % 1 == 0:
                train_writer.add_scalar(
                    Path(args.path).stem,
                    loss.item(),
                    idx
                    + iteration
                    * (len(training_set) + len(cross_validation_set)),
                )

            if idx % args.visualize_step_size == 0:
                mask_generator.save(args.path)

            training_losses.append(loss.item())
            optimizer.step()
            optimizer.zero_grad()

            if any(fid % args.visualize_step_size == 0 for fid in fids):
                # save the model
                mask_generator.save(args.path)
                # visualize
                if args.visualize:
                    maxid = np.argmax(
                        [fid % args.visualize_step_size == 0 for fid in fids]
                    ).item()
                    visualize(
                        maxid,
                        fids,
                        sr_image,
                        mask_slice,
                        target,
                        saliency,
                        "train",
                    )

        mean_training_loss = torch.tensor(training_losses).mean()
        logger.info("Average training loss: %.3f", mean_training_loss.item())

        """
            Cross validation
        """

        mask_generator.eval()

        progress_bar = tqdm(
            total=len(cross_validation_set),
            desc=f"Iteration {iteration} on cross validation set",
        )

        cross_validation_losses = []

        for idx, data in enumerate(cross_validation_loader):

            progress_bar.update(args.batch_size)

            try:
                fids, names, sr_image, o_image, target, _ = unzip_data(data, saliency)
            except ValueError:
                continue
          
            # inference
            with torch.no_grad():

                mask_slice = mask_generator(o_image.cuda())
                # loss = get_loss(mask_slice, target.cuda(), thresh_list.cuda())
                loss = get_loss(mask_slice, target.cuda())

            if idx % 1 == 0:
                cross_writer.add_scalar(
                    Path(args.path).stem,
                    loss.item(),
                    idx
                    + iteration
                    * (len(training_set) + len(cross_validation_set))
                    + len(training_set),
                )

            if any(fid % args.visualize_step_size == 0 for fid in fids):
                if args.visualize:
                    maxid = np.argmax(
                        [fid % args.visualize_step_size == 0 for fid in fids]
                    ).item()
                    visualize(
                        maxid,
                        fids,
                        sr_image,
                        mask_slice,
                        target,
                        saliency,
                        "cross",
                    )

            cross_validation_losses.append(loss.item())

        mean_cross_validation_loss = (
            torch.tensor(cross_validation_losses).mean().item()
        )
        logger.info(
            "Average cross validation loss: %.3f", mean_cross_validation_loss
        )

        """
            Finalize one ieteration
        """
        if mean_cross_validation_loss < mean_cross_validation_loss_before:
            mask_generator.save(args.path + ".best")
            overfitting_counter = 0
        else:
            overfitting_counter += 1

        if overfitting_counter >= 3:
            return

        mean_cross_validation_loss_before = min(
            mean_cross_validation_loss_before, mean_cross_validation_loss
        )

        # mask_generator.save(args.path + ".iter%d" % iteration)

        # check if we need to reduce learning rate.
        scheduler.step(mean_cross_validation_loss)

        """
            Test, only when the overfitting_counter is 0
        """
        if overfitting_counter == 0:

            progress_bar = tqdm(
                total=len(test_set),
                desc=f"Iteration {iteration} on test set",
            )

            test_losses = []

            for idx, data in enumerate(test_loader):

                progress_bar.update(args.batch_size)

                # # extract data from dataloader

                try:
                    fids, names, sr_image, o_image, target, _ = unzip_data(
                        data, saliency
                    )
                except ValueError:
                    continue

                # inference
                with torch.no_grad():

                    mask_slice = mask_generator(o_image.cuda())
                    # loss = get_loss(mask_slice, target.cuda(), thresh_list.cuda())
                    loss = get_loss(mask_slice, target.cuda())

                if any(fid % 1 == 0 for fid in fids):
                    if args.visualize:
                        maxid = np.argmax([fid % 1 == 0 for fid in fids]).item()
                        visualize(
                            maxid,
                            fids,
                            sr_image,
                            mask_slice,
                            target,
                            saliency,
                            "test",
                        )

                test_losses.append(loss.item())

            mean_test_loss = torch.tensor(test_losses).mean().item()
            logger.info(
                "Average test loss: %.3f", mean_test_loss,
            )


if __name__ == "__main__":

    # set the format of the logger
    coloredlogs.install(
        fmt="%(asctime)s [%(levelname)s] %(name)s:%(funcName)s[%(lineno)s] -- %(message)s",
        level="INFO",
    )

    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        help="The path to store the generator parameters.",
        required=True,
    )
    parser.add_argument(
        "--init",
        type=str,
        help="The path to init the generator parameters.",
        default="",
    )
    parser.add_argument(
        "--log", type=str, help="The logging file.", required=True,
    )
    parser.add_argument(
        "-g",
        "--ground_truth",
        type=str,
        help="The ground truth file.",
        required=True,
    )
    # parser.add_argument('-o', '--output', type=str,
    #                     help='The output name.', required=True)
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        help="The confidence score threshold for calculating accuracy.",
        default=0.7,
    )
    parser.add_argument(
        "--gt_confidence_threshold",
        type=float,
        help="The confidence score threshold for calculating accuracy.",
        default=0.7,
    )
    parser.add_argument(
        "--maskgen_file",
        type=str,
        help="The file that defines the neural network.",
        required=True,
    )
    parser.add_argument(
        "--iou_threshold",
        type=float,
        help="The IoU threshold for calculating accuracy in object detection.",
        default=0.5,
    )
    parser.add_argument(
        "--saliency_threshold",
        type=float,
        help="The threshold to binarize the saliency.",
        default=0.5,
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        help="Number of iterations for optimizing the mask.",
        default=500,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Number of iterations for optimizing the mask.",
        default=2,
    )
    parser.add_argument(
        "--app", type=str, help="The name of the model.", required=True,
    )
    parser.add_argument(
        "--tile_size", type=int, help="The tile size of the mask.", default=8
    )
    parser.add_argument(
        "--learning_rate", type=float, help="The learning rate.", default=1e-4
    )
    parser.add_argument(
        "--gamma",
        type=float,
        help="The gamma parameter for focal loss.",
        default=2,
    )
    parser.add_argument(
        "--visualize", type=bool, help="Visualize the heatmap.", default=False
    )
    parser.add_argument(
        "--local_rank",
        default=-1,
        type=int,
        help="The GPU id for distributed training",
    )
    parser.add_argument(
        "--visualize_step_size",
        default=-1,
        type=int,
        help="The step size for training visualization",
    )
    parser.add_argument(
        "--architecture",
        default="vgg11",
        type=str,
        help="The backbone architecture",
    )

    parser.add_argument(
        "--num_workers",
        default=5,
        type=int,
        help="Number of workers for data loading",
    )

    parser.add_argument(
        "--test_set", required=True, type=str, help="Test set",
    )
    parser.add_argument(
        "--training_set", required=True, type=str, help="Training set",
    )
    parser.add_argument(
        "--no_class_check", dest="class_check", action="store_false"
    )
    parser.set_defaults(class_check=True)

    args = parser.parse_args()

    main(args)
