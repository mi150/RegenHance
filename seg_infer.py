import cv2
import tensorrt as trt
import torch
import pycuda.autoinit
import pycuda.driver as cuda
import warnings
import torch.nn.functional as F
import time
from torchvision.ops import box_convert,nms
from cv2.dnn import NMSBoxes
import numpy as np
import pickle
cfx = cuda.Device(0).make_context()
warnings.filterwarnings("ignore", category=Warning)
all=0

class seg_infer():

    def __init__(self,args,cfx):
        #model_path=args
        model_path = args.detect_model_path
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
        self.context.set_binding_shape(0, (1, 3, 1080, 1920))


    # 推理
    def inference(self,imgs):
        global all
        h_output=[]
        labels = torch.empty(19,1080,1920).cuda().int()
        for stream in range(imgs.shape[0]):
            for index in range(imgs.shape[1]):
                img=imgs[stream][index].unsqueeze(0).float()
                # array = img[0].cpu().numpy()
                # array = array.transpose(1, 2, 0)
                #
                # # Convert numpy array to BGR color format expected by OpenCV
                # image = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
                #
                # # Save image to disk using OpenCV
                #
                # cv2.imwrite("SR_image/%04d.png" % all, image)
                all+=1
                img=img/255
                # 模型预测
                self.cfx.push()
                self.context.execute_async(bindings=[int(img.data_ptr()), int(labels.data_ptr())], stream_handle=self.stream.handle)
                self.cfx.pop()
                self.stream.synchronize()
                label=torch.argmax(labels,dim=0).cpu()
                #print(label)
                #while 1:pass
                self.write_results(label,stream,index+self.gop_index*imgs.shape[1])
        self.gop_index+=1
        return h_output

    def inference_single(self,img):
        global all
        h_output=[]
        img = torch.from_numpy(img).cuda().float()
        labels = torch.empty(19,1080,1920).cuda().float()
        print(img.shape)
        # array = img[0].cpu().numpy()
        # array = array.transpose(1, 2, 0)

        # Convert numpy array to BGR color format expected by OpenCV
        #image = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)

        # Save image to disk using OpenCV

        #cv2.imwrite("SR_image/%04d.png" % all, image)
        #all+=1
        #img=img/255
        t1=time.time()
        # 模型预测
        self.cfx.push()
        self.context.execute_async(bindings=[int(img.data_ptr()), int(labels.data_ptr())], stream_handle=self.stream.handle)
        self.cfx.pop()
        self.stream.synchronize()
        label=torch.argmax(labels,dim=0).cpu()
        # print(label)
        # while 1:pass
        #print(label)
        #while 1:pass
        self.write_results(label,0,all)
        all+=1
        #print(time.time()-t1)
        self.gop_index+=1
        return h_output




    def write_results(self,outputs,stream,index):
        fname="seg_results/neuro/result_str_neuro_60_"+str(stream)+"_"+str(index)+".txt"
        file = open(fname, 'wb')
        pickle.dump(outputs, file,-1)
        file.close()

if __name__ == "__main__":

    seg=seg_infer('hardnet.engine',cfx)

    for i in range(10):
        file_path = '/home/dodo/trt_test/profile/nemo_seg/video/chunk%04d/1080p_per_neuro_sr_pngs_60/' % i
        for j in range(120):
            print(file_path+"%04d.png"%j)
            img=cv2.imread(file_path+"%04d.png"%j)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            value_scale = 255
            mean = [0.406, 0.456, 0.485]
            mean = [item * value_scale for item in mean]
            std = [0.225, 0.224, 0.229]
            std = [item * value_scale for item in std]
            img = (img - mean) / std
            img = np.transpose(np.array([img], dtype="float32"), (0, 3, 1, 2))
            #img = img / 255.0
            seg.inference_single(img)