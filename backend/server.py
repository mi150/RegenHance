import os
import logging
import cv2 as cv
from dds_utils import (Results, Region, extract_images_from_video)
from .object_detector import Detector

import torch
from model import Model
from data import common
from .utility import (checkpoint, quantize)
from .SR import SRDetector
from munch import *
import yaml
from PIL import Image
import importlib
from torchvision.transforms import transforms
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
import time
import numpy as np


def split_lr_then_sr(net, device, lr, scale, shave):
    lr = lr.squeeze(0)
    h, w = lr.size()[1:]
    h_half, w_half = int(h / 2), int(w / 2)
    h_chop, w_chop = h_half + shave, w_half + shave

    # split large image to 4 patch to avoid OOM error
    lr_patch = torch.FloatTensor(4, 3, h_chop, w_chop)
    lr_patch[0].copy_(lr[:, 0:h_chop, 0:w_chop])
    lr_patch[1].copy_(lr[:, 0:h_chop, w - w_chop:w])
    lr_patch[2].copy_(lr[:, h - h_chop:h, 0:w_chop])
    lr_patch[3].copy_(lr[:, h - h_chop:h, w - w_chop:w])
    lr_patch = lr_patch.to(device)

    # run refine process in here!
    sr = net(lr_patch, scale).data

    h, h_half, h_chop = h * scale, h_half * scale, h_chop * scale
    w, w_half, w_chop = w * scale, w_half * scale, w_chop * scale

    # merge splited patch images
    result = torch.FloatTensor(3, h, w).to(device)
    result[:, 0:h_half, 0:w_half].copy_(sr[0, :, 0:h_half, 0:w_half])
    result[:, 0:h_half, w_half:w].copy_(sr[1, :, 0:h_half, w_chop - w + w_half:w_chop])
    result[:, h_half:h, 0:w_half].copy_(sr[2, :, h_chop - h + h_half:h_chop, 0:w_half])
    result[:, h_half:h, w_half:w].copy_(sr[3, :, h_chop - h + h_half:h_chop, w_chop - w + w_half:w_chop])
    sr = result.unsqueeze(0)
    return sr


class Server:
    """The server component of DDS protocol. Responsible for running DNN
       on low resolution images, tracking to find regions of interest and
       running DNN on the high resolution regions of interest"""

    def __init__(self, config, nframes=None):
        self.config = config

        self.logger = logging.getLogger("server")
        handler = logging.NullHandler()
        self.logger.addHandler(handler)
        self.curr_fid = 0
        self.nframes = nframes
        self.last_requested_regions = None
        self.logger.info("Server started")
        self.detector = Detector()
        self.edsr_time = 0
        self.infer_time = 0
        # 不在初始化这里读入yml的原因：gt文件也需要初始化，但是工作目录在workspace下面；server端的工作目录在dds主文件下面，
        # 在网络环境下运行的时候，初始化server是单独的工作目录；但是gt模式的时候是公用一个工作目录workspace/
        # self.SRconfig = self.load_configuration('backend/SRconfiguration.yml')
        # self.ASRconfig=self.load_configuration("backend/ASRconfiguration.yml")
        self.train_flag = False

    def load_configuration(self, filename):  # SR 初始化
        """read configuration information from yaml file

        Returns:
            dict: information of the yaml file
        """
        with open(filename, 'r') as config:
            config_info = yaml.load(config, Loader=yaml.FullLoader)
        # use munch to provide class-like accessment to python dictionary
        args = munchify(config_info)
        return args

    def reset_state(self, nframes):
        self.curr_fid = 0
        self.nframes = nframes
        self.last_requested_regions = None
        for f in os.listdir("server_temp"):
            os.remove(os.path.join("server_temp", f))
        for f in os.listdir("server_temp-cropped"):
            os.remove(os.path.join("server_temp-cropped"), f)

    def perform_server_cleanup(self):
        for f in os.listdir("server_temp"):
            os.remove(os.path.join("server_temp", f))
        for f in os.listdir("server_temp-cropped"):
            os.remove(os.path.join("server_temp-cropped", f))

    def simulate_low_query(self, start_fid, end_fid, images_direc,
                           results_dict, simulation=True,
                           rpn_enlarge_ratio=0.0, extract_regions=True):
        if extract_regions:
            # If called from actual implementation
            # This will not run
            base_req_regions = Results()
            for fid in range(start_fid, end_fid):
                base_req_regions.append(
                    Region(fid, 0, 0, 1, 1, 1.0, 2,
                           self.config.high_resolution))
            extract_images_from_video(images_direc, base_req_regions)

        batch_results = Results()

        self.logger.info(f"Getting results with threshold "
                         f"{self.config.low_threshold} and "
                         f"{self.config.high_threshold}")
        # Extract relevant results
        for fid in range(start_fid, end_fid):
            fid_results = results_dict[fid]  # f  no results  ,this place error
            for single_result in fid_results:
                single_result.origin = "low-res"
                batch_results.add_single_result(
                    single_result, self.config.intersection_threshold)

        detections = Results()
        rpn_regions = Results()
        # Divide RPN results into detections and RPN regions
        for single_result in batch_results.regions:
            if (single_result.conf > self.config.prune_score and
                    single_result.label == "vehicle"):
                detections.add_single_result(
                    single_result, self.config.intersection_threshold)
            else:
                rpn_regions.add_single_result(
                    single_result, self.config.intersection_threshold)

        regions_to_query = self.get_regions_to_query(rpn_regions, detections)

        return detections, regions_to_query

    def perform_detection(self, images_direc, resolution, fnames=None, images=None):
        #  results=self.perform_detection("server_temp",self.config.low_resolution,fnames)
        #  gt 模式需要用到
        final_results = Results()
        if fnames is None:
            fnames = sorted(os.listdir(images_direc))
        self.logger.info(f"Running inference on {len(fnames)} frames")
        infer_time = 0
        for fname in fnames:
            if "png" not in fname:
                continue
            fid = int(fname.split(".")[0])
            image = None
            if images:
                image = images[fid]
            else:
                image_path = os.path.join(images_direc, fname)
                image = cv.imread(image_path)
            self.logger.info(f"{fid} {self.config.method} image size:{image.shape[1]},{image.shape[0]}")
            # 目标检测需要输入rgb格式
            t1 = time.time()
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            detection_results = self.detector.infer(image)            
            t2 = time.time()
            infer_time = infer_time + t2 - t1
            # (label, conf, box_tuple)
            # print(detection_results)
            frame_with_no_results = True
            for label, conf, (x, y, w, h) in detection_results:
                r = Region(fid, x, y, w, h, conf, label,
                           resolution, origin="mpeg")
                # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
                final_results.append(r)
                frame_with_no_results = False

            # 检测的时候如果没有检测到结果就加入"no obj"
            if frame_with_no_results:
                final_results.append(
                    Region(fid, 0, 0, 0, 0, 0.1, "no obj", resolution))
        print("gt average infer time per frame:", infer_time / self.config.batch_size)
        f = open(f"{self.config.video_oname}_gt_record.txt", "a+")
        f.write(f"{infer_time / self.config.batch_size}\n")
        f.close()
        return final_results

    def perform_detection_cubic(self, images_direc, resolution, fnames=None, images=None):
        #  results=self.perform_detection("server_temp",self.config.low_resolution,fnames)
        final_results = Results()
        if fnames is None:
            fnames = sorted(os.listdir(images_direc))
        self.logger.info(f"Running inference on {len(fnames)} frames")
        for fname in fnames:
            if "png" not in fname:
                continue
            fid = int(fname.split(".")[0])
            image = None
            if images:
                image = images[fid]
            else:
                image_path = os.path.join(images_direc, fname)
                image = cv.imread(image_path)
            nw = int(image.shape[1] / self.config.low_resolution)
            nh = int(image.shape[0] / self.config.low_resolution)
            cubic_img = cv.resize(image, (nw, nh), fx=0, fy=0,
                                  interpolation=cv.INTER_CUBIC)

            self.logger.info(f"{fid} {self.config.method} image size:{image.shape[1]},{image.shape[0]};"
                             f"{cubic_img.shape[1]},{cubic_img.shape[0]}")

            # 目标检测需要输入rgb格式
            cubic_img = cv.cvtColor(cubic_img, cv.COLOR_BGR2RGB)
            detection_results = self.detector.infer(cubic_img)
            # (label, conf, box_tuple)
            # print(detection_results)
            frame_with_no_results = True
            for label, conf, (x, y, w, h) in detection_results:
                r = Region(fid, x, y, w, h, conf, label,
                           resolution, origin="mpeg")
                # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
                final_results.append(r)
                frame_with_no_results = False

            # 检测的时候如果没有检测到结果就加入"no obj"
            if frame_with_no_results:
                final_results.append(
                    Region(fid, 0, 0, 0, 0, 0.1, "no obj", resolution))

        return final_results

    def perform_detection_lr(self, images_direc, resolution, fnames=None, images=None):
        #  results=self.perform_detection("server_temp",self.config.low_resolution,fnames)
        final_results = Results()
        if fnames is None:
            fnames = sorted(os.listdir(images_direc))
        self.logger.info(f"Running inference on {len(fnames)} frames")
        infer_time = 0
        for fname in fnames:
            if "png" not in fname:
                continue
            fid = int(fname.split(".")[0])
            image = None
            if images:
                image = images[fid]
            else:
                image_path = os.path.join(images_direc, fname)
                image = cv.imread(image_path)
            self.logger.info(f"{fid} {self.config.method} image size:{image.shape[1]},{image.shape[0]}")
            # 目标检测需要输入rgb格式
            t1 = time.time()
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            detection_results = self.detector.infer(image)
            t2 = time.time()
            infer_time = infer_time + t2 - t1
            # (label, conf, box_tuple)
            # print(detection_results)
            frame_with_no_results = True
            for label, conf, (x, y, w, h) in detection_results:
                r = Region(fid, x, y, w, h, conf, label,
                           resolution, origin="mpeg")
                # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
                final_results.append(r)
                frame_with_no_results = False

            # 检测的时候如果没有检测到结果就加入"no obj"
            if frame_with_no_results:
                final_results.append(
                    Region(fid, 0, 0, 0, 0, 0.1, "no obj", resolution))
        print("low res average infer time per frame++++++++++++++++++++++", infer_time / self.config.batch_size)
        f = open(f"{self.config.video_oname}_lr_record.txt", "a+")
        f.write(f"{infer_time / self.config.batch_size}\n")
        f.close()
        return final_results

    def perform_detection_edsr(self, images_direc, resolution, fnames=None, images=None):
        #  results=self.perform_detection("server_temp",self.config.low_resolution,fnames)
        # SR 初始化
        self.SRconfig = self.load_configuration('backend/SRconfiguration.yml')
        self.SRconfig.scale = [int(1 / self.config.low_resolution)]
        print('self.SRconfig', self.SRconfig)

        final_results = Results()
        if fnames is None:
            fnames = sorted(os.listdir(images_direc))
        self.logger.info(f"Running inference on {len(fnames)} frames")

        t1 = time.time()

        #  SR model prepare==========================================
        self.logger.info('SR model prepare===========================================================')
        torch.manual_seed(self.SRconfig.seed)
        ckp = checkpoint(self.SRconfig)
        global SRmodel
        SRmodel = Model(self.SRconfig, ckp)
        t = SRDetector(self.SRconfig, SRmodel, ckp)
        torch.set_grad_enabled(False)
        t.model.eval()
        #  SR model prepare===============================================
        t2 = time.time()
        print("sr model prepare:", t2 - t1)
        sr_time = 0
        infer_time = 0
        for fname in fnames:
            if "png" not in fname:
                continue
            fid = int(fname.split(".")[0])
            src_image = None
            if images:
                src_image = images[fid]
            else:
                image_path = os.path.join(images_direc, fname)
                src_image = cv.imread(image_path)
            # SR ======================================================
            t3 = time.time()
            image, = common.set_channel(src_image, n_channels=self.SRconfig.n_colors)
            image, = common.np2Tensor(image, rgb_range=self.SRconfig.rgb_range)
            image, = t.prepare(image.unsqueeze(0))
            sr = t.model(image, idx_scale=0)
            sr = quantize(sr, self.SRconfig.rgb_range).squeeze(0)
            normalized = sr * 255 / self.SRconfig.rgb_range
            ndarr = normalized.byte().permute(1, 2, 0).cpu().numpy()
            t4 = time.time()
            sr_time = sr_time + t4 - t3
            # # save 影响我出刀的速度
            # if not os.path.exists('server_EDSR'):
            #     os.mkdir('server_EDSR')
            # cv.imwrite(os.path.join("server_EDSR", f'{fname}'),ndarr)
            # # end SR=============================================

            self.logger.info(f"{fid} {self.config.method} image size:{src_image.shape[1]},{src_image.shape[0]};"
                             f"{ndarr.shape[1]},{ndarr.shape[0]}")

            # 目标检测要输入rgb格式
            t5 = time.time()
            ndarr = cv.cvtColor(ndarr, cv.COLOR_BGR2RGB)
            detection_results = self.detector.infer(ndarr)
            t6 = time.time()
            infer_time = infer_time + t6 - t5
            # (label, conf, box_tuple)
            # print(detection_results)
            frame_with_no_results = True
            for label, conf, (x, y, w, h) in detection_results:
                r = Region(fid, x, y, w, h, conf, label,
                           resolution, origin="mpeg")
                # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
                final_results.append(r)
                frame_with_no_results = False

            # 检测的时候如果没有检测到结果就加入"no obj"
            if frame_with_no_results:
                final_results.append(
                    Region(fid, 0, 0, 0, 0, 0.1, "no obj", resolution))

        sr_time = sr_time + t2 - t1
        print("average per frame time,edsr ,  infer", sr_time / self.config.batch_size,
              infer_time / self.config.batch_size)
        f = open(f"{self.config.video_oname}_edsr_record.txt", "a+")
        f.write(f"{sr_time / self.config.batch_size}    {infer_time / self.config.batch_size}\n")
        f.close()
        return final_results

    def perform_detection_asr(self, images_direc, resolution, fnames=None, images=None):
        #  results=self.perform_detection("server_temp",self.config.low_resolution,fnames)
        # SR 初始化
        self.ASRconfig = self.load_configuration("backend/ASRconfiguration.yml")
        self.ASRconfig.scale = int(1 / self.config.low_resolution)
        print("当前工作目录", os.getcwd())
        print('self.ASRconfig', self.ASRconfig)

        final_results = Results()
        if fnames is None:
            fnames = sorted(os.listdir(images_direc))
        self.logger.info(f"Running inference on {len(fnames)} frames")
        time_before_prepare_asr_model = time.time()
        #  SR model prepare==========================================
        # 一个函数运行需要根据不同项目的配置，动态导入对应的配置文件运行
        module = importlib.import_module(f"model.{self.ASRconfig.model}")
        net = module.Net(multi_scale=False,
                         scale=self.ASRconfig.scale,
                         group=self.ASRconfig.group)
        # 将python对象编码成Json字符串
        # var  将对象变成属性：属性值字典
        state_dict = torch.load(self.ASRconfig.ckpt_path, map_location='cpu')
        if list(state_dict.items())[0][0][:7] != "module.":
            net.load_state_dict(state_dict, strict=True)
        else:
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove "module."
                new_state_dict[name] = v
            net.load_state_dict(new_state_dict, strict=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 如torch能检测到cuda，就用cuda
        net = net.to(device)  # 把模型放上去
        net = nn.DataParallel(net, device_ids=range(self.ASRconfig.num_gpu))
        time_after_prepare_asr_model = time.time()
        time_of_prepare_asr_model = time_after_prepare_asr_model - time_before_prepare_asr_model
        print(f"time of prepare asr model : {time_of_prepare_asr_model}")
        with open(f"{self.config.video_oname}_carn_record.txt", "a+") as f:
            f.write(f"time of prepare asr model : {time_of_prepare_asr_model}\n")
        #  SR model prepare===============================================
        time_of_asr_one_batch = 0
        time_of_infer_one_batch = 0
        for fname in fnames:
            if "png" not in fname:
                continue
            fid = int(fname.split(".")[0])
            src_image = None
            if images:
                src_image = images[fid]
            else:
                image_path = os.path.join(images_direc, fname)
                src_image = Image.open(image_path)
            # SR ======================================================
            time_before_asr_one_frame = time.time()
            lr = src_image.convert("RGB")
            lr = transforms.Compose([
                transforms.ToTensor()
            ])(lr)
            lr = lr.unsqueeze(0).to(device)
            # sr = net(lr, self.ASRconfig.scale)
            sr = split_lr_then_sr(net=net, device=device, lr=lr, scale=self.ASRconfig.scale, shave=self.ASRconfig.shave)
            sr = sr.detach().squeeze(0).cpu()
            ndarr = sr.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
            time_after_asr_one_frame = time.time()
            time_of_asr_one_batch = time_of_asr_one_batch + time_after_asr_one_frame - time_before_asr_one_frame
            # # save 影响我出刀的速度
            # if not os.path.exists('server_ASR'):
            #     os.mkdir('server_ASR')
            # im=Image.fromarray(ndarr)
            # im.save(os.path.join("server_ASR", f'{fname}'))
            # # end SR=============================================

            self.logger.info(f"{fid} {self.config.method} image size:{src_image.size[0]},{src_image.size[1]};"
                             f"{ndarr.shape[1]},{ndarr.shape[0]}")
            # PIL 图片size[0]是宽

            # 目标检测要输入rgb格式，asr模型是PIL读入，输出的是RGB格式
            # ndarr = cv.cvtColor(ndarr, cv.COLOR_BGR2RGB)
            time_before_infer_one_frame = time.time()
            detection_results = self.detector.infer(ndarr)
            time_after_infer_one_frame = time.time()
            time_of_infer_one_batch = time_of_infer_one_batch + time_after_infer_one_frame - time_before_infer_one_frame
            # (label, conf, box_tuple)
            # print(detection_results)
            frame_with_no_results = True
            for label, conf, (x, y, w, h) in detection_results:
                r = Region(fid, x, y, w, h, conf, label,
                           resolution, origin="mpeg")
                # 从服务器传到client端都origin会重新命名为low-res， 函数get_first_phase_results
                final_results.append(r)
                frame_with_no_results = False

            # 检测的时候如果没有检测到结果就加入"no obj"
            if frame_with_no_results:
                final_results.append(
                    Region(fid, 0, 0, 0, 0, 0.1, "no obj", resolution))

        print("average per frame time of asr : ", time_of_asr_one_batch / self.config.batch_size)
        print("average per frame time of infer : ", time_of_infer_one_batch / self.config.batch_size)
        with open(f"{self.config.video_oname}_carn_record.txt", "a+") as f:
            f.write(f"average per frame time of asr : {time_of_asr_one_batch / self.config.batch_size}\n")
            f.write(f"average per frame time of infer : {time_of_infer_one_batch / self.config.batch_size}\n")
        return final_results

    def perform_low_query(self, vid_data):
        # Write video to file
        with open(os.path.join("server_temp", "temp.mp4"), "wb") as f:
            f.write(vid_data.read())

        # 从server_temp/temp.mp4中取出图片
        start_fid = self.curr_fid
        end_fid = min(self.curr_fid + self.config.batch_size, self.nframes)
        self.logger.info(f"Processing frames from {start_fid} to {end_fid - 1}")
        req_regions = Results()
        for fid in range(start_fid, end_fid):  # 左包右不包
            req_regions.append(Region(fid, 0, 0, 1, 1, 1.0, 2, self.config.low_resolution))
        # extract_images_from_video("server_temp",req_regions) 的参数req_regions主要是看帧号
        # 如果是第二轮传输不是每一帧都有req_region的话就要把视频帧和原第一轮传输的帧对应起来
        time_before_decode = time.time()
        extract_images_from_video("server_temp", req_regions)
        time_after_decode = time.time()
        time_of_decode_one_batch = time_after_decode - time_before_decode
        self.logger.info(f"average per frame time of decode : {time_of_decode_one_batch / self.config.batch_size}\n")
        with open(f"{self.config.video_oname}_carn_record.txt", "a+") as f:
            f.write(f"average per frame time of decode : {time_of_decode_one_batch / self.config.batch_size}\n")
        fnames = [f for f in os.listdir("server_temp") if "png" in f]
        fnames.sort()

        if self.config.method == "edsr":
            self.logger.info(f"{self.config.method}++++++++++++++++++++++++++++++++++++++++++++++++")
            results = self.perform_detection_edsr("server_temp", self.config.low_resolution, fnames)
        elif self.config.method == "asr":
            self.logger.info(f"{self.config.method}++++++++++++++++++++++++++++++++++++++++++++++++")
            results = self.perform_detection_asr("server_temp", self.config.low_resolution, fnames)
        elif self.config.method == 'cubic':
            self.logger.info(f"{self.config.method}++++++++++++++++++++++++++++++++++++++++++++++++")
            results = self.perform_detection_cubic("server_temp", self.config.low_resolution, fnames)
        else:
            self.logger.info("not find proper baseline in config!!!")
        self.curr_fid = end_fid  # refresh curfid
        detections_list = []
        for r in results.regions:
            detections_list.append(
                [r.fid, r.x, r.y, r.w, r.h, r.conf, r.label])
        # 传回给client端的是字典，其中value是列表[[r.fid, r.x, r.y, r.w, r.h, r.conf, r.label],[r.fid, r.x, r.y, r.w, r.h, r.conf, r.label] ... ]
        self.perform_server_cleanup()

        # if start_fid == 30:
        #     self.train_flag = True
        # else:
        #     self.train_flag = False
        self.train_flag = False
        return {"results": detections_list, "whether_train": self.train_flag}

    def perform_online_training(self, vid_data, gt_vid_data):
        self.logger.info(f"Ready to process online training!")
        if not os.path.exists("server_temp_lr"):
            os.makedirs("server_temp_lr")
        if not os.path.exists("server_temp_gt"):
            os.makedirs("server_temp_gt")
        with open(os.path.join("server_temp_lr", "temp.mp4"), "wb") as f:
            f.write(vid_data.read())
        with open(os.path.join("server_temp_gt", "temp.mp4"), "wb") as f:
            f.write(gt_vid_data.read())

        # 从server_temp/temp.mp4中取出图片
        start_fid = self.curr_fid
        end_fid = min(self.curr_fid + self.config.batch_size, self.nframes)
        self.logger.info(f"Processing frames from {start_fid} to {end_fid - 1}")
        req_regions = Results()
        for fid in range(start_fid, end_fid):  # 左包右不包
            req_regions.append(Region(fid, 0, 0, 1, 1, 1.0, 2, self.config.low_resolution))
        # extract_images_from_video("server_temp",req_regions) 的参数req_regions主要是看帧号
        # 如果是第二轮传输不是每一帧都有req_region的话就要把视频帧和原第一轮传输的帧对应起来
        extract_images_from_video("server_temp_lr", req_regions)
        extract_images_from_video("server_temp_gt", req_regions)
        fnames = [f for f in os.listdir("server_temp_lr") if "png" in f]
        fnames.sort()
        gt_fnames = [f for f in os.listdir("server_temp_gt") if "png" in f]
        gt_fnames.sort()

        if self.config.method == "edsr":
            self.logger.info(f"{self.config.method}++++++++++++++++++++++++++++++++++++++++++++++++")
            results = self.online_training_edsr("server_temp_lr", "server_temp_gt", self.config.low_resolution, fnames)
        elif self.config.method == "asr":
            self.logger.info(f"{self.config.method}++++++++++++++++++++++++++++++++++++++++++++++++")
            results = self.online_training_asr("server_temp_lr", "server_temp_gt", self.config.low_resolution, fnames)
        else:
            self.logger.info("not find proper baseline in config!!!")
        self.curr_fid = end_fid  # refresh curfid
        return {"results": results}

    def online_training_asr(self, images_direc, gt_images_direc, resolution, fnames=None, images=None):
        self.logger.info(f"Ready to process online training of asr!")
        # SR 初始化
        self.ASRconfig = self.load_configuration("backend/ASRconfiguration.yml")
        self.ASRconfig.scale = int(1 / self.config.low_resolution)
        print("当前工作目录", os.getcwd())
        print('self.ASRconfig', self.ASRconfig)

        final_results = Results()
        if fnames is None:
            fnames = sorted(os.listdir(images_direc))
        self.logger.info(f"Running online training on {len(fnames)} frames")
        t1 = time.time()
        #  SR model prepare==========================================
        # 一个函数运行需要根据不同项目的配置，动态导入对应的配置文件运行
        module = importlib.import_module(f"model.{self.ASRconfig.model}")
        net = module.Net(multi_scale=False,
                         scale=self.ASRconfig.scale,
                         group=self.ASRconfig.group)
        # 将python对象编码成Json字符串
        # var  将对象变成属性：属性值字典
        state_dict = torch.load(self.ASRconfig.ckpt_path)
        if list(state_dict.items())[0][0][:7] != "module.":
            net.load_state_dict(state_dict, strict=True)
        else:
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove "module."
                new_state_dict[name] = v
            net.load_state_dict(new_state_dict, strict=True)

        my_optim = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), self.ASRconfig.lr)
        loss_fn = nn.L1Loss()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = net.to(device)
        net = nn.DataParallel(net, device_ids=range(2))
        net.train()
        #  SR model prepare===============================================
        t2 = time.time()
        print("prepare sr model time : ", t2 - t1)
        net.train()
        sr_time = 0
        lr_images = []
        hr_images = []
        for fname in fnames:
            lr_image = cv.imread(os.path.join(images_direc, fname))
            lr_image = cv.resize(lr_image, None, fx=1/2, fy=1/2, interpolation=cv.INTER_AREA)
            lr_image = cv.cvtColor(lr_image, cv.COLOR_BGR2RGB)
            lr_images.append(lr_image)
            hr_image = cv.imread(os.path.join(gt_images_direc, fname))
            hr_image = cv.resize(hr_image, None, fx=1/2, fy=1/2, interpolation=cv.INTER_AREA)
            hr_image = cv.cvtColor(hr_image, cv.COLOR_BGR2RGB)
            hr_images.append(hr_image)
        t3 = time.time()
        for epoch in range(self.ASRconfig.num_epochs):
            for i, (lr, hr) in enumerate(zip(lr_images, hr_images)):
                hr = np.transpose(hr / 255.0, [2, 0, 1])
                lr = np.transpose(lr / 255.0, [2, 0, 1])
                hr = torch.from_numpy(hr).type(torch.FloatTensor).to(device)
                lr = torch.from_numpy(lr).type(torch.FloatTensor).to(device)
                hr = hr.unsqueeze(0)
                lr = lr.unsqueeze(0)
                sr = net(lr, self.ASRconfig.scale)
                loss = loss_fn(sr, hr)
                my_optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), self.ASRconfig.clip)
                my_optim.step()
            self.logger.info(f"===============================================epoch {epoch} down! loss = {loss}")

        t4 = time.time()
        sr_time = sr_time + t4 - t3

        torch.save(net.state_dict(), self.ASRconfig.ckpt_path)
        sr_time = sr_time + t2 - t1
        print("===============================================online training time : ", sr_time)
        f = open(f"{self.config.video_oname}_carn_record.txt", "a+")
        f.write(f"===============================================online training time : {sr_time}\n")
        f.close()
        return True

    def online_training_edsr(self, images_direc, gt_images_direc, resolution, fnames=None, images=None):
        pass

    def whether_train(self):
        return {"whether_train": self.train_flag}
