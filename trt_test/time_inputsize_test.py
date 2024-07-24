import tensorrt as trt
import torch
import pickle
import numpy as np
import pycuda.driver as cuda
import time
import cv2
import os
import pycuda.autoinit
import copy
import warnings

# 忽略TensorRT的警告
warnings.filterwarnings("ignore", category=Warning)
# 初始化(创建引擎，为输入输出开辟&分配显存/内存.)

def init():
    sumtime=0
    t_to_cuda_time=0
    infer_time=0
    t_to_cpu_time=0
    #model_path = "./output_0_2/mb2_model.engine"
    model_path ="./DETR/detr-s.engine"
    #model_path="/home/ubuntu/csw/AccMPEG/VideoAnalysis/mi_dds_sr/yolov5/yolov5s.engine"
    # 加载runtime，记录log
    runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
    # 反序列化模型
    engine = runtime.deserialize_cuda_engine(open(model_path, "rb").read())
    print("输入",engine.get_binding_shape(0))
    print("输出",engine.get_binding_shape(1))
    # 1. Allocate some host and device buffers for inputs and outputs:
    print(trt.volume(engine.get_binding_shape(0)))
    h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=trt.nptype(trt.float32))
    h_output = cuda.pagelocked_empty(abs(trt.volume(engine.get_binding_shape(1))), dtype=trt.nptype(trt.int32))    
    # Allocate device memory for inputs and outputs.
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    # Create a stream in which to copy inputs/outputs and run inference.
    stream = cuda.Stream()
    # 推理上下文
    context = engine.create_execution_context()
    return context, h_input, h_output, stream, d_input, d_output, sumtime, t_to_cuda_time, infer_time, t_to_cpu_time

# 推理
def inference(data_path):
    global context, h_input, h_output, stream, d_input, d_output, sumtime, t_to_cuda_time, infer_time, t_to_cpu_time
    load_normalized_data(data_path, h_input)
    t0 = time.time()
    # 将图片数据送到cuda显存中
    cuda.memcpy_htod_async(d_input, h_input, stream)
    stream.synchronize()
    t1 = time.time()
    print("togpu时间", t1 - t0)
    t_to_cuda_time=t_to_cuda_time+t1-t0
    # 模型预测
    t1 = time.time()
    context.execute_async(batch_size=1,bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
    stream.synchronize()
    t2 = time.time()
    print("推理时间", t2 - t1)
    infer_time=infer_time+t2-t1
    # 将结果送回内存中
    t2 = time.time()
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    
    ## 异步等待结果
    stream.synchronize()
    t3 = time.time()
    # Return the host output.
    #t3 = time.time()
    print("tocpu'时间", t3 - t2)
    t_to_cpu_time=t_to_cpu_time+t3-t2
    sumtime = sumtime + t2 - t1
    print(h_output.shape)
    #print(np.max(h_output), np.min(h_output))
    #h_output = np.reshape(h_output, (360, 640))
    return h_output

# 加载数据并将其喂入提供的pagelocked_buffer中.
def load_normalized_data(data_path, pagelocked_buffer, target_size=(1333, 800)):
    #img = cv2.imread(data_path)
    img=np.random.rand(800,1333,3)
    #img = cv2.resize(img, target_size, cv2.INTER_AREA)
    img = cv2.resize(img, target_size, cv2.INTER_CUBIC)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #img = img / 255.0
    img = np.transpose(np.array([img], dtype="float32"), (0, 3, 1, 2))
    #print("???")
    #print(img)
    #print([img for _ in range(128)])
    #img=np.concatenate([img for _ in range(128)],0)
    print(img.shape)
    # 此时img.shape为H * W * C: 360, 640, 3
    # print("图片shape", img.shape)
    # Flatten the image into a 1D array, normalize, and copy to pagelocked memory.
    np.copyto(pagelocked_buffer, img.ravel())

if __name__ == '__main__':
    context, h_input, h_output, stream, d_input, d_output, sumtime, t_to_cuda_time, infer_time, t_to_cpu_time = init()
    imgs_path = "data/video_test/decode/"
    #imgs_path = "/home/ubuntu/csw/AccMPEG/VideoAnalysis/mi_dds_sr/Faster-RCNN/image_src/"
    #imgs_path = "/home/liangm/video_analysis/det_image_src/"
    #imgs = os.listdir(imgs_path)
    pred = [[] for _ in range(3000)]
    for i in range(500):
        print(i)
        #img_path = os.path.join(imgs_path, img)
        output = inference(data_path="")
        print(output.shape)
        copy_tmp = copy.deepcopy(output)
        #copy_tmp = np.argmax(copy_tmp, axis=0)
        #tmp = torch.from_numpy(copy_tmp).mul(1).clamp(0,255).round().div(1).detach().squeeze(0).byte().numpy()
        #res_img = tmp.transpose(1, 2, 0)
        #i = int(img_path.replace('.jpg','').replace('data/video_test/decode/frame-',''))
        #i = int(img.split('.')[0])
        #print(i)
        #####pred[i]=copy_tmp
        #cv2.imwrite('/home/ubuntu/csw/AccMPEG/VideoAnalysis/mi_dds_sr/Faster-RCNN/SR_part_image1/%010d.png'%i, res_img)
        #print("type  output:", type(copy_tmp), "output.shape:", copy_tmp.shape, "output:", copy_tmp, "max:", np.amax(copy_tmp), "\n")
    print("用时:", sumtime)
    print(t_to_cuda_time)
    print(infer_time)
    print(t_to_cpu_time)
    mask_ = np.array(pred)
    #mask_ = mask.astype(np.int8)
    #print(np.amax(mask_), np.amin(mask_))
    #print(mask_)
    #mask = torch.from_numpy(mask_)
    #mask = mask.softmax(dim=1)[:,1:2,:,:]
    #print(mask_.shape)
    #torch.save(mask.to(torch.device('cpu')),"mask_AccSR_pixel_720_trt.pth")
    #with open('yanshou.txt', 'wb') as file:
    #    pickle.dump(mask_, file, protocol = 4)

#0.012+0.002+0.005=0.020
