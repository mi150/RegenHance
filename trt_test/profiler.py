from openvino.runtime import Core
import pycuda.driver as cuda

import tensorrt as trt
import torch
import logging
import time
import yaml
import numpy as np
import cpuinfo
import GPUtil
import os
import cv2
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pynvml as nvml
import multiprocessing
torch.cuda.empty_cache()
def has_integrated_gpu():
    #return False
    cpu_data = cpuinfo.get_cpu_info()

    cpu_brand = cpu_data.get('brand_raw', '').lower()
    if 'intel' in cpu_brand:
        if 'core' in cpu_brand or 'pentium' in cpu_brand or 'celeron' in cpu_brand:
            return True
    elif 'amd' in cpu_brand:
        if 'ryzen' in cpu_brand or 'athlon' in cpu_brand or 'a-series' in cpu_brand:
            return True
    return False


def get_gpu_count():
    gpus = GPUtil.getGPUs()
    return len(gpus)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class MyGpuAllocator(trt.IGpuAllocator):
    def __init__(self, gpu_index):
        super().__init__()
        self.gpu_index = gpu_index

    def allocate(self, memory_size):
        cuda.Device(self.gpu_index).make_context()
        return cuda.mem_alloc(memory_size)

    def free(self, memory):
        cuda.Device(self.gpu_index).make_context()
        memory.free()


class Profiler():
    def __init__(self,config):
        
        #multiprocessing.set_start_method('spawn')
        
        self.device_list={"CPU":-1,"IGPU":-1,"GPU":-1}
        self.config=config
        #self.handle=nvml.nvmlDeviceGetHandleByIndex(0)
        
        self.profiler_yaml_path=config["profiler_yaml_path"]
        self.width = config["video_reslution"][0]
        self.height=config["video_reslution"][1]
        cuda.init()
        self.gpu_count =2
        logging.info("GPU_count:%i",self.gpu_count)
        self.cfx=cuda.Device(0).make_context()

    def get_GPU_usage(self,flag,lock):
        usage=[-1]
        nvml.nvmlInit() 
        
        handle=nvml.nvmlDeviceGetHandleByIndex(0)
        #nvml.nvmlSetUpdateInterval(handle,20)
        while True:
            with lock:
                if flag.value==0:
                    break
            #if self.util.gpu>0:
            util=nvml.nvmlDeviceGetUtilizationRates(handle)
            usage.append(util.gpu)
            #logging.info("GPU Utilization: %i",util.gpu)
        logging.info("GPU Utilization: %i",max(usage))
    def profile_model(self):
        '''
        #profile mask
        if len(self.config["mask_model_path"]["mask_onnx_path"]) > 0:
            if has_integrated_gpu():
                self.profile_onnx(self.config["mask_model_path"]["mask_onnx_path"],"GPU",self.config["mask_model_path"]["shape"])
            self.profile_onnx(self.config["mask_model_path"]["mask_onnx_path"], "CPU",self.config["mask_model_path"]["shape"])
        
        #cuda device
        for i in range(self.gpu_count):
            self.profile_trt(self.config["mask_model_path"]["mask_trt_path"],"GPU",i,self.config["mask_model_path"]["shape"])

        #profile sr
        if len(self.config["sr_model_path"]["sr_onnx_path"])>0:
            if has_integrated_gpu():
                self.profile_onnx(self.config["sr_model_path"]["sr_onnx_path"],"GPU",self.config["sr_model_path"]["shape"])
            #self.profile_onnx(self.config["sr_model_path"]["sr_onnx_path"], "CPU",self.config["sr_model_path"]["shape"])
        #cuda device
        '''
        for i in range(self.gpu_count):
            self.profile_trt(self.config["sr_model_path"]["sr_trt_path"],"GPU",i,self.config["sr_model_path"]["shape"])
        '''
        #profile detect
        if len(self.config["detect_model_path"]["detect_onnx_path"])>0:
            if has_integrated_gpu():
                self.profile_onnx(self.config["detect_model_path"]["detect_onnx_path"],"GPU",self.config["detect_model_path"]["shape"])
            self.profile_onnx(self.config["detect_model_path"]["detect_onnx_path"], "CPU",self.config["detect_model_path"]["shape"])
        # cuda device
        
        for i in range(self.gpu_count):
            self.profile_trt(self.config["detect_model_path"]["detect_trt_path"],"GPU",i,self.config["detect_model_path"]["shape"])
        '''

    def profile_onnx(self,model_onnx,device,shape):
        logging.info("Begin to profile %s on %s", model_onnx,device)
        ie = Core()
        compiled_model_onnx = ie.compile_model(model=model_onnx, device_name=device)
        output_layer_onnx = compiled_model_onnx.output(0)
        img=np.random.rand(1,3,shape[0],shape[1])
        batch_size=1
        #[batch_size,timecost]
        profile_list=list()
        
        while True:
  
            #try:
            logging.info("batchsize:%i",batch_size)
            _ = compiled_model_onnx([img])[output_layer_onnx]
            start=time.time()
            for _ in range(5):
                # img=cv2.imread("/home/dodo/trt_test/profile/yolov5-master/data/images/zidane.jpg")
                # img=cv2.resize(img,(360,640), interpolation = cv2.INTER_AREA)
                output = compiled_model_onnx(np.transpose(np.array([img]), (0, 3, 1, 2)))[output_layer_onnx]
                array = output[0].transpose(1, 2, 0)
                #cv2.imwrite("image.png", array)
                while 1:pass
            end=time.time()
            if (end-start)/5>1:
                logging.warning("Time limit is exceeded!")
                break
            profile_list.append([batch_size,(end-start)/5])
            batch_size=batch_size*2
            img = np.concatenate((img, img), axis=0)
            #except:
                #logging.warning("The batchsize is too large!")
                #break
        with open(self.profiler_yaml_path, 'a', encoding='utf-8') as f:
            yaml.dump(data={"profile_"+model_onnx+"_"+(device if device=="CPU" else "IGPU"):profile_list}, stream=f, allow_unicode=True)
        logging.info("Finish profiling %s on %s", model_onnx, (device if device=="CPU" else "IGPU"))

    def profile_trt(self,model_trt,device,selected_gpu_index,shape):
        logging.info("Begin to profile %s on %s:%i", model_trt, device,selected_gpu_index)
        #init trt
        gpu_allocator=MyGpuAllocator(selected_gpu_index)
        model_path = model_trt
        #
        runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
        #cuda.Device(0).use()
        engine = runtime.deserialize_cuda_engine(open(model_path, "rb").read())

        context = engine.create_execution_context()
        stream = cuda.Stream()
        profile_list=list()
        batch_size=1

        while 1:
            logging.info("batchsize:%i",batch_size)
            context.set_binding_shape(0, (batch_size, 3, shape[0], shape[1]))
            imgs = [torch.rand(batch_size, 3, shape[0], shape[1]).cuda().float() for _ in range(5)]
            #print(img.shape)
            output_datas = [torch.rand(batch_size, shape[4], shape[2], shape[3]).cuda().float() for _ in range(5)]

            #warmup
            #self.cfx.push()
            #context.execute_async_v2(bindings=[img.data_ptr(), output_datas[0].data_ptr()], stream_handle=stream.handle)
            #self.cfx.pop()
            #profile
            flag=multiprocessing.Value('i',1)
            lock=multiprocessing.Lock()
            start=time.time()
            multiprocessing.Process(target=self.get_GPU_usage,args=(flag,lock)).start()

            for img,output_data in zip(imgs,output_datas):
                for _ in range(50):
                    #stream.synchronize()
                    #print(img)
                    self.cfx.push()
                    context.execute_async_v2(bindings=[int(img.data_ptr()), int(output_data.data_ptr())],
                                             stream_handle=stream.handle)
                    self.cfx.pop()
                    stream.synchronize()
                    # print(output_data.to('cpu'))
            with lock:
                flag.value=0
            end=time.time()
            # usage
            print(end-start)
            #p.terminate()
            profile_list.append([batch_size,(end-start)/250])
            batch_size=batch_size*2
            if batch_size>8:
                logging.warning("The batchsize is too large!")
                break
            elif (end-start)/250>1:
                logging.warning("Time limit is exceeded!")
                break
            #except:
                #logging.warning("The batchsize is too large!")
                #break
        #yaml
        self.cfx.pop()
        with open(self.profiler_yaml_path, 'a', encoding='utf-8') as f:
            yaml.dump(data={"profile_" + model_trt + "_" + device: profile_list}, stream=f, allow_unicode=True)
        #empty cache
        torch.cuda.empty_cache()
        logging.info("Finish profiling %s on %s:%i", model_trt, device, selected_gpu_index)
if __name__=="__main__":
    
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
    #print(has_integrated_gpu())
    profiler=Profiler(config)
    profiler.profile_model()

    
