from dds_utils import (Results, read_results_dict, evaluate, cleanup, Region,
                       compute_regions_size, merge_boxes_in_results, extract_images_from_video,calc_iou,filter_bbox_group)
import os
import logging
import torch
import cv2 as cv
from backend.object_detector import Detector
import time
import torchvision as tv


final_results = Results()
detector=Detector()
torch.set_grad_enabled(False)
images_direc = '/home/ubuntu/csw/AccMPEG/VideoAnalysis/mi_dds_sr/Faster-RCNN/SR_part_image1/'
#images_direc='/home/ubuntu/VideoAnalytics/mi_dds_sr/Faster-RCNN/video_test/src1/'
fnames = sorted(os.listdir(images_direc))
sumtime=0
for fname in fnames:

    if "png" not in fname:
        continue

    
    fid = int(fname.split(".")[0])

    
    image = None
    #if fid>10:
    #    break
    image_path = os.path.join(images_direc, fname)
    image = cv.imread(image_path)
    #image=cv.resize(image, (480, 270))
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    
    t1=time.time()

    detection_results = detector.infer(image)

    # (label, conf, box_tuple)
    # print(detection_results)
    frame_with_no_results = True
    for label, conf, (x, y, w, h) in detection_results:
        r = Region(fid, x, y, w, h, conf, label,
                   1, origin="mpeg")

        final_results.append(r)
        frame_with_no_results = False

    if frame_with_no_results:
        final_results.append(
            Region(fid, 0, 0, 0, 0, 0.1, "no obj", 1))
    if (fid+1)%100==0:
        print('detect fid',fid)
    t2=time.time()
    sumtime = sumtime+t2-t1
print("Detect time:", sumtime)
final_results.write('video_part_results_360_3_01')
 
