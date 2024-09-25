"""
    Compress the video through gradient-based optimization.
"""
import cv2
import argparse
import gc
import importlib
import logging
import time
from pathlib import Path
import math

import coloredlogs
import enlighten
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import io
from tqdm import tqdm

from dnn.dnn_factory import DNN_Factory
from utilities.bbox_utils import center_size
from utilities.compressor import h264_roi_compressor_segment
from utilities.loss_utils import focal_loss as get_loss
from utilities.mask_utils import *
from utilities.results_utils import read_ground_truth, read_results
from utilities.timer import Timer
from utilities.video_utils import get_qp_from_name, read_videos, write_video
from utilities.visualize_utils import (
    visualize_dist_by_summarywriter,
    visualize_heat_by_summarywriter,
)

thresh_list = [0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8]

sns.set()


def main(args):
    torch.cuda.empty_cache()
    gc.enable()

    # initialize
    logger = logging.getLogger("blackgen")
    logger.addHandler(logging.FileHandler("blackgen.log"))
    torch.set_default_tensor_type(torch.FloatTensor)

    # read the video frames (will use the largest video as ground truth)
    videos, bws, video_names = read_videos(args.inputs, logger, sort=True)

    # construct applications
    app = DNN_Factory().get_model(args.app)

    maskgen_spec = importlib.util.spec_from_file_location(
        "maskgen", args.maskgen_file
    )
    maskgen = importlib.util.module_from_spec(maskgen_spec)
    maskgen_spec.loader.exec_module(maskgen)
    mask_generator = maskgen.FCN()
    mask_generator.load(args.path)
    mask_generator.eval()
    mask_generator.cuda()

    cached_images = []

    # construct the mask
    mask_shape = [
        len(videos[-1]),  ## video total frames number 3000
        1,
        720 // args.tile_size,
        #math.ceil(1080/args.tile_size),
        1280 // args.tile_size,
        #math.ceil(1920/args.tile_size),
    ]
    mask = torch.ones(mask_shape).float()
    
    # construct the writer for writing the result
    writer = SummaryWriter(f"runs/{args.app}/{args.output}")
    print("start time")

    for temp in range(1):

        logger.info(f"Processing application")
        progress_bar = enlighten.get_manager().counter(
            total=len(videos[-1]), desc=f"{app.name}", unit="frames"
        )

        # application.cuda()

        losses = []
        f1s = []
        sumtime=0
        for fid, (video_slices, mask_slice) in enumerate(
            zip(zip(*videos), mask.split(1))
        ):

            progress_bar.update()
            o_images_direc='/data01/wangweijun/accsr/AccMPEG/origin_img/'
            o_images_direc='/data01/wangweijun/accsr/AccMPEG/test_temp_pic/'
            o_image = io.read_image(o_images_direc+"%010d.png"%fid)/255.0
            image_tmp = o_image.permute(1,2,0).numpy()*255.0
            en_image = cv2.resize(image_tmp, (720, 1080), fx=0, fy=0) / 255.0
            in_image = T.ToTensor()(en_image)

            hq_image = in_image.unsqueeze(dim=0)
            print(hq_image)
            print(hq_image.shape)
            print(fid)

            t1 = time.time()
            # construct hybrid image
            with torch.no_grad():
                # gt_result = application.inference(hq_image.cuda(), detach=True)[0]
                # _, _, boxes, _ = application.filter_results(
                #     gt_result, args.confidence_threshold
                # )
                # boxes = center_size(boxes)

                # size1 = boxes[:, 2] * boxes[:, 3]
                # sum1s.append(size1.sum())
                # boxes[:, 2:] = boxes[:, 2:] + 7 * args.tile_size
                # size2 = boxes[:, 2] * boxes[:, 3]
                # sum2s.append(size2.sum())
                # # ratios.append(size2.sum() / size1.sum())
                # mask_slice[:, :, :, :] = generate_mask_from_regions(
                #     mask_slice, boxes, 0, args.tile_size
                # )

                # mask_gen = mask_generator(
                #     torch.cat([hq_image, hq_image - lq_image], dim=1).cuda()
                # )
                hq_image = hq_image.cuda()
                #print(hq_image.shape)
                # mask_generator = mask_generator.cpu()
                # with Timer("maskgen", logger):
                mask_gen = mask_generator(hq_image)
                #print(mask_gen.shape)
                # losses.append(get_loss(mask_gen, ground_truth_mask[fid]))
                mask_gen = mask_gen.softmax(dim=1)[:, 1:2, :, :]
                #print(mask_gen.shape)
                #mask_gen = mask_gen[:,1:2,:,:]
                #print(mask_gen.shape)
                # mask_lb = dilate_binarize(mask_gen, args.bound, args.conv_size)
                # mask_ub = dilate_binarize(mask_gen, args.upper_bound, args.conv_size)
                mask_slice[:, :, :, :] = mask_gen
                # mask_slice[:, :, :, :] = torch.where(mask_gen > 0.5, torch.ones_like(mask_gen), torch.zeros_like(mask_gen))
                #print(mask_slice.shape)
            t2=time.time()
            sumtime=sumtime+t2-t1
            print(t2-t1)
            # visualization
            if fid % args.visualize_step_size == 0:

                image = T.ToPILImage()(video_slices[-1][0, :, :, :])
                cached_images.append(image)

                mask_slice = mask_slice.detach().cpu()

                writer.add_image("raw_frame", video_slices[-1][0, :, :, :], fid)

                visualize_heat_by_summarywriter(
                    image, mask_slice, "inferred_saliency", writer, fid, args,
                )

                visualize_dist_by_summarywriter(
                    mask_slice, "saliency_dist", writer, fid,
                )

                mask_slice = sum(
                    [(mask_slice > thresh).float() for thresh in thresh_list]
                )

                visualize_heat_by_summarywriter(
                    image, mask_slice, "binarized_saliency", writer, fid, args,
                )
            
        logger.info("In video %s", args.output)
        logger.info("The average loss is %.3f" % torch.tensor(losses).mean())

        # application.cpu()

    mask.requires_grad = False
    print("end time")
    print(sumtime)
    torch.save(mask.to(torch.device('cpu')), "importance.pth")

if __name__ == "__main__":

    # set the format of the logger
    coloredlogs.install(
        fmt="%(asctime)s [%(levelname)s] %(name)s:%(funcName)s[%(lineno)s] -- %(message)s",
        level="INFO",
    )

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--app", type=str, help="The name of the model.", required=True,
    )

    parser.add_argument(
        "-i",
        "--inputs",
        nargs="+",
        help="The video file names. The largest video file will be the ground truth.",
        required=True,
    )
    parser.add_argument(
        "-g",
        "--ground_truth",
        help="The video file names. The largest video file will be the ground truth.",
        type=str,
        required=True,
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "-s",
        "--source",
        type=str,
        help="The original video source.",
        required=True,
    )
    # parser.add_argument('-g', '--ground_truth', type=str, help='The ground truth results.', required=True)
    parser.add_argument(
        "-o", "--output", type=str, help="The output name.", required=True
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        help="The confidence score threshold for calculating accuracy.",
        default=0.7,
    )
    parser.add_argument(
        "--iou_threshold",
        type=float,
        help="The IoU threshold for calculating accuracy in object detection.",
        default=0.5,
    )
    parser.add_argument(
        "--tile_size", type=int, help="The tile size of the mask.", default=8
    )
    parser.add_argument(
        "--maskgen_file",
        type=str,
        help="The file that defines the neural network.",
        required=True,
    )
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        help="The path of pth file that stores the generator parameters.",
        required=True,
    )
    # parser.add_argument(
    #     "--upper_bound", type=float, help="The upper bound for the mask", required=True,
    # )
    # parser.add_argument(
    #     "--lower_bound", type=float, help="The lower bound for the mask", required=True,
    # )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--bound", type=float, help="The lower bound for the mask",
    )
    action.add_argument(
        "--perc", type=float, help="The percentage of modules to be encoded."
    )
    parser.add_argument(
        "--smooth_frames",
        type=int,
        help="Proposing one single mask for smooth_frames many frames",
        default=10,
    )
    parser.add_argument(
        "--visualize_step_size",
        type=int,
        help="Proposing one single mask for smooth_frames many frames",
        default=100,
    )
    parser.add_argument("--conv_size", type=int, default=1)
    parser.add_argument("--hq", type=int, default=-1)
    parser.add_argument("--lq", type=int, default=-1)

    # parser.add_argument('--mask', type=str,
    #                     help='The path of the ground truth video, for loss calculation purpose.', required=True)

    args = parser.parse_args()

    main(args)
