from PIL import Image, ImageDraw
import os
import os
from datetime import datetime
from munch import *
import yaml
import cv2 as cv
import torch
import torch.nn.functional as F
import torchvision as tv
from utility import (checkpoint, quantize)
from SR import SRDetector
from model import Model
#from data import common
# import numpy as np
# import time
# from multiprocessing import Pool
# from dds_utils import (Results, read_results_dict, evaluate, cleanup, Region,
#                        compute_regions_size, merge_boxes_in_results, extract_images_from_video, calc_iou,
#                        filter_bbox_group)
# from skimage.measure import label, regionprops


def load_configuration():
    with open("config.yml", 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # use munch to provide class-like accessment to python dictionary
    args = munchify(config)
    return args

if __name__ == "__main__":
    


    
    en = 0
    # SR configuration===============================================================
    SRconfig = load_configuration()
    # print("config",SRconfig)
    #  SR model prepare==========================================

    print('SR model prepare-------------------------------')
    torch.manual_seed(SRconfig.seed)
    ckp = checkpoint(SRconfig)
    global SRmodel

    SRmodel = Model(SRconfig, ckp)
    while 1: pass
    t = SRDetector(SRconfig, SRmodel, ckp)
    torch.set_grad_enabled(False)
    t.model.eval()

#     onnxfile = 'EDSR.onnx'
#     batch_size = 1
#     input_shape = (3, 360, 640)
#     image_sample = torch.randn(batch_size, *input_shape).cuda()
#     idx_scale_sample = torch.Tensor([0])
#     x = (image_sample, idx_scale_sample)
#
#     dynamic_axes = {"input": {0: "batch"}, "output": {0: "batch"}}
#     dynamic_axes = {
#     'input': {0: 'batch_size', 2: "height", 3: 'width'},  # 这么写表示NCHW都会变化
# }
#     torch.onnx.export(t.model, x, onnxfile, input_names=['input'], output_names=['output'], opset_version=11, do_constant_folding=True,export_params=True)
#     print("export ONNX file success!")
#
