import tensorrt as trt
import torch
import pickle
import numpy as np
import pycuda
import pycuda.driver as cuda
import time
from skimage.measure import label, regionprops
from scipy.ndimage import  zoom
from scipy.special import softmax
import cv2
import warnings
from openvino.runtime import Core
# 忽略TensorRT的警告
warnings.filterwarnings("ignore", category=Warning)

# 初始化(创建引擎，为输入输出开辟&分配显存/内存.)
class trt_infer():

    def __init__(self,args,cfx,device):
        self.cfx = cfx
        self.args=args
        model_path = args.mask_model_path
        # 加载runtime，记录log
        self.device=device
        self.thres=args.thres
        if self.device=='CUDA':
            self.runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
            # 反序列化模型
            self.engine = self.runtime.deserialize_cuda_engine(open(model_path, "rb").read())

            # 1. Allocate some host and device buffers for inputs and outputs:
            self.h_input = cuda.pagelocked_empty(abs(trt.volume(self.engine.get_binding_shape(0))), dtype=trt.nptype(trt.float32))
            self.h_output = cuda.pagelocked_empty(abs(trt.volume(self.engine.get_binding_shape(1))), dtype=trt.nptype(trt.float32))

            # Allocate device memory for inputs and outputs.
            self.d_input = cuda.mem_alloc(self.h_input.nbytes)
            self.d_output = cuda.mem_alloc(self.h_output.nbytes)
            # Create a stream in which to copy inputs/outputs and run inference.
            self.stream = cuda.Stream()
            # 推理上下文
            self.context = self.engine.create_execution_context()
        #self.context.set_binding_shape(0,(1,3,360,640))
        else:
            self.onnx_path = args.onnx_path
            # Create inference session
            self.ie = Core()
            self.ie.set_property(device_name="CPU", properties={"CPU_THREADS_NUM": 32})

            self.model_onnx = self.ie.read_model(model=self.onnx_path)
            #UHD
            #self.uhd_compiled_model_onnx = self.ie.compile_model(model=self.model_onnx, device_name="GPU")
            #self.uhd_output_layer_onnx = self.uhd_compiled_model_onnx.output(0)
            #CPU
            self.cpu_compiled_model_onnx = self.ie.compile_model(model=self.model_onnx, device_name="CPU",config={"CPU_THREADS_NUM": 32})
            self.cpu_output_layer_onnx = self.cpu_compiled_model_onnx.output(0)


    # 推理
    def inference(self,img):
        if self.device=="CUDA":
            #img=img.flatten()
            #img=torch.zeros(691200,)
            #input_data=np.array(img.cpu())
            #np.copyto(self.h_input, input_data)
            #self.cfx.push()
            self.load_normalized_data(img, self.h_input)
            cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
            self.context.execute_async(bindings=[int(self.d_input), int(self.d_output)], stream_handle=self.stream.handle)
            # 将结果送回内存中
            cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
            #self.cfx.pop()
            ## 异步等待结果
            self.stream.synchronize()
            # Return the host output.

            self.h_output = np.reshape(self.h_output, (360, 640))
            return self.h_output
            
            #accmpeg
            h_output = np.reshape(self.h_output, (1,2,23,40))
            h_output=softmax(h_output,axis=1)
            
            return h_output[0,1]
        elif self.device=="CPU":
            #img=self.load_normalized_data(img)
            #img = img / 255.0
            if self.args.seg:

                img=zoom(img,zoom=(1,1,2,2),order=0)
                t1 = time.time()
                h_output = self.cpu_compiled_model_onnx([img])[self.cpu_output_layer_onnx]
                h_output=h_output[:,0,:22,:]
                #print(time.time() - t1)
                return h_output
            t1 = time.time()
            h_output = self.cpu_compiled_model_onnx([img])[self.cpu_output_layer_onnx]
            #print(time.time() - t1)
            h_output=zoom(h_output[0],zoom=(22/360,40/640),order=0)

            return h_output
        else:
            h_output = self.uhd_compiled_model_onnx([img])[self.uhd_output_layer_onnx]
            return h_output
    # 加载数据并将其喂入提供的pagelocked_buffer中.
    @staticmethod
    def load_normalized_data(device,img, pagelocked_buffer=None, target_size=(640, 360)):
        img = cv2.resize(img, target_size, cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0
        img = np.transpose(np.array([img], dtype="float32"), (0, 3, 1, 2))
        if device=="CUDA":
            np.copyto(pagelocked_buffer, img.ravel())
        else:
            return img




