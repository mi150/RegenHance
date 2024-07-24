import os
import logging
import torch
from .backbone import resnet50_fpn_backbone
from .network_files import FasterRCNN
import cv2 as cv
import time
class Detector:
    classes={
        "vehicle": [3, 6, 7, 8],
        "persons": [1, 2, 4],
        "roadside-objects": [10, 11, 13, 14]

    }
    #  3': 'car', '6': 'bus''7': 'train'   '8': 'truck'
    # '1': 'person', '2': 'bicycle', '4': 'motorcycle',
    # '10': 'traffic light', '11': 'fire hydrant', '13': 'stop sign', '14': 'parking meter',
    def __init__(self, num_classes=91,train_weights="fasterrcnn_resnet50_fpn_coco.pth"):
        self.logger = logging.getLogger("object_detector")
        handler = logging.NullHandler()
        self.logger.addHandler(handler)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # create model
        t1=time.time()
        self.backbone=resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d)
        self.model=FasterRCNN(backbone=self.backbone,num_classes=num_classes,rpn_score_thresh=0.5)
        self.model.load_state_dict(torch.load(train_weights, map_location=device))
        self.model.to(device)
        self.model.eval()
        t2=time.time()
        self.logger.info(f"end create model{t2-t1}")
        #---end create model

    def infer(self,image):
        # image 格式：rgb,  numpy : h  w  c
        # 需要预处理
        width=image.shape[1]
        height=image.shape[0]
        # c  h  w  的tensor
        img_tensor = torch.from_numpy(image / 255.).permute(2, 0, 1).float().cuda()
        # 增加一个维度
        img = torch.unsqueeze(img_tensor, dim=0)

        with torch.no_grad():
            predictions = self.model(img)[0]
            boxes = predictions["boxes"].to("cpu").numpy()
            labels = predictions["labels"].to("cpu").numpy()
            scores = predictions["scores"].to("cpu").numpy()

            # print(boxes)
            # print(labels)
            # print(scores)
            results = []
            # print(boxes.shape[0])  #100
            for i in range(boxes.shape[0]):  # boxes  列表  n*4
                label_number=labels[i].item()  # int
                relevant_class=False
                # 看能不能找到relevant class
                for j in Detector.classes.keys():  # 找到
                    if label_number in Detector.classes[j]:
                        label=j
                        relevant_class=True
                        break
                if not relevant_class:
                    continue
                # 坐标以图片左上角为坐标原点
                x1, y1, x2, y2 = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
                # print(x1, y1, x2, y2,"=======================bbox")
                box_tuple = (x1 / width, y1 / height, (x2 - x1) / width, (y2 - y1) / height)
                conf = scores[i]
                results.append((label, conf, box_tuple))

        return results

















