import time

import cv2
import tensorrt as trt
import torch
import pycuda.autoinit
import warnings
import torch.nn.functional as F
import numpy as np
from trt_test.dds_utils import Results,Region
import pycuda.driver as cuda
cfx = cuda.Device(0).make_context()
# 忽略TensorRT的警告
warnings.filterwarnings("ignore", category=Warning)
class_names = [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush' ]
classes = {
    "vehicle": [2, 5, 6, 7],
    "persons": [0, 1, 3],
    "roadside-objects": [9, 10, 12, 13]

}
all=0
# 初始化(创建引擎，为输入输出开辟&分配显存/内存.)
class detect_infer():

    def __init__(self,args,cfx):
        model_path = args.detect_model_path
        #model_path = args
        self.cfx= cfx
        logger = trt.Logger(trt.Logger.WARNING)
        logger.min_severity = trt.Logger.Severity.ERROR
        self.runtime = trt.Runtime(logger)
        # 加载runtime，记录log
        #self.runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
        trt.init_libnvinfer_plugins(logger, '')
        # 反序列化模型
        self.engine = self.runtime.deserialize_cuda_engine(open(model_path, "rb").read())
        # 1. Allocate some host and device buffers for inputs and outputs:
        # self.h_input = cuda.pagelocked_empty(trt.volume(self.engine.get_binding_shape(0)), dtype=trt.nptype(trt.float32))
        #self.h_output = cuda.pagelocked_empty(trt.volume(self.engine.get_binding_shape(1)), dtype=trt.nptype(trt.float32))
        # # Allocate device memory for inputs and outputs.
        # self.d_input = cuda.mem_alloc(self.h_input.nbytes)
        # self.d_output = cuda.mem_alloc(self.h_output.nbytes)
        # Create a stream in which to copy inputs/outputs and run inference.
        self.stream = cuda.Stream()
        self.gop_index=0
        # 推理上下文
        self.context = self.engine.create_execution_context()
        self.context.set_binding_shape(0, (1, 3, 1088, 1920))


    # 推理
    def inference(self,imgs):
        global all
        h_output=[]
        num= torch.empty(1).cuda().int()
        boxes=torch.empty(100,4).cuda().float()
        scores = torch.empty(100).cuda().float()
        labels = torch.empty(100).cuda().int()
        for stream in range(imgs.shape[0]):
            for index in range(imgs.shape[1]):
                t1=time.time()
                img=imgs[stream][index].unsqueeze(0).float()
                img=F.interpolate(img,size=(img.shape[2]+8, img.shape[3]), mode='bilinear', align_corners=False)
                # if stream == 5:
                # array = img[0].cpu().numpy()
                # array = array.transpose(1, 2, 0)
                #
                # # Convert numpy array to BGR color format expected by OpenCV
                # image = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
                #
                # # Save image to disk using OpenCV
                #
                # cv2.imwrite("puzzle_img/image%09d.png" % all, image)
                # all+=1
                img=img/255
                # 模型预测
                self.cfx.push()
                self.context.execute_async(bindings=[int(img.data_ptr()), int(num.data_ptr()), int(boxes.data_ptr())
                                                     , int(scores.data_ptr()), int(labels.data_ptr())], stream_handle=self.stream.handle)
                self.cfx.pop()
                self.stream.synchronize()
                #print(boxes)
                outputs=(np.array(num.cpu()), np.array(boxes.cpu()), np.array(scores.cpu()), np.array(labels.cpu()))
                self.nms(outputs,stream,index+self.gop_index*imgs.shape[1])
                #print(time.time()-t1)
        self.gop_index+=1
        return h_output


    def inference_single(self,img,index):
        global all
        img=torch.from_numpy(img).cuda()
        h_output=[]
        num= torch.empty(1).cuda().int()
        boxes=torch.empty(100,4).cuda().float()
        scores = torch.empty(100).cuda().float()
        labels = torch.empty(100).cuda().int()

        # img=img.unsqueeze(0).float()

        img=F.interpolate(img,size=(img.shape[2]+8, img.shape[3]), mode='bilinear', align_corners=False)
        # print(img.shape)
        # while 1: pass
        # if stream == 5:
        # array = img[0].cpu().numpy()
        # array = array.transpose(1, 2, 0)
        #
        # # Convert numpy array to BGR color format expected by OpenCV
        # image = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
        #
        # # Save image to disk using OpenCV
        #
        # cv2.imwrite("puzzle_img/image%09d.png" % all, image)
        # all+=1
        #img=img/255
        # 模型预测
        self.cfx.push()
        self.context.execute_async(bindings=[int(img.data_ptr()), int(num.data_ptr()), int(boxes.data_ptr())
                                             , int(scores.data_ptr()), int(labels.data_ptr())], stream_handle=self.stream.handle)
        self.cfx.pop()
        self.stream.synchronize()
        #print(boxes)
        outputs=(np.array(num.cpu()), np.array(boxes.cpu()), np.array(scores.cpu()), np.array(labels.cpu()))
        self.nms(outputs,0,index)
        #self.gop_index+=1
        #return h_output


    def nms(self,outputs,stream,index):
        num,boxes,scores,labels=outputs
        results=Results()
        for i in range(100):

            label_number = labels[i]  # int
            relevant_class = False
            # 看能不能找到relevant class
            for j in classes.keys():  # 找到
                if label_number in classes[j]:
                    label = j
                    relevant_class = True
                    break
            if not relevant_class:
                continue
            # 坐标以图片左上角为坐标原点
            x1, y1, x2, y2 = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
            # print(x1, y1, x2, y2,"=======================bbox")
            box_tuple = (x1 / 1920, y1 / 1088, (x2 - x1) / 1920, (y2 - y1) / 1088)
            if x2==x1 or y2==y1:
                continue
            conf = scores[i]

            results.append(Region(index,box_tuple[0],box_tuple[1],box_tuple[2],box_tuple[3],conf,label,0,origin='mpeg'))
        results.write('results/result_js_2_%d'%stream)

        #return results

if __name__ == "__main__":

    detect=detect_infer('yolov5s.engine',cfx)

    for i in range(25):
        file_path = '/home/dodo/trt_test/profile/nemo/video/video/chunk%04d/1080p_per_neuro_sr_pngs_60/' % i
        for j in range(120):
            print(file_path+"%04d.png"%j)
            img=cv2.imread(file_path+"%04d.png"%j)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255.0
            img = np.transpose(np.array([img], dtype="float32"), (0, 3, 1, 2))

            detect.inference_single(img,i*120+j)