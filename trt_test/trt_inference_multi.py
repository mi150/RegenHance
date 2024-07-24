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
import multiprocessing 

# 初始化(创建引擎，为输入输出开辟&分配显存/内存.)

def init(model_name):
    sumtime=0
    model_path = model_name
    # 加载runtime，记录log
    runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
    # 反序列化模型
    engine = runtime.deserialize_cuda_engine(open(model_path, "rb").read())
    #print("输入",engine.get_binding_shape(0))
    #print("输出",engine.get_binding_shape(1))
    # 1. Allocate some host and device buffers for inputs and outputs:
    h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=trt.nptype(trt.float32))
    h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=trt.nptype(trt.int32))    
    # Allocate device memory for inputs and outputs.
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    # Create a stream in which to copy inputs/outputs and run inference.
    stream = cuda.Stream()
    # 推理上下文
    context = engine.create_execution_context()
    return context, h_input, h_output, stream, d_input, d_output, sumtime



# 加载数据并将其喂入提供的pagelocked_buffer中.
def load_normalized_data(data_path, pagelocked_buffer, target_size,batch_size):
    img = cv2.imread(data_path)
    #img = cv2.resize(img, target_size, cv2.INTER_AREA)
    img = cv2.resize(img, target_size, cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #img = img / 255.0
    img = np.transpose(np.array([img], dtype="float32"), (0, 3, 1, 2))
    if batch_size>1:
        img=np.concatenate([img for _ in range(batch_size)],0)
    #print(img)
    # 此时img.shape为H * W * C: 360, 640, 3
    # print("图片shape", img.shape)
    # Flatten the image into a 1D array, normalize, and copy to pagelocked memory.
    np.copyto(pagelocked_buffer, img.ravel())

def one_infer(pid,model_name,target_size,batch_size):
    context, h_input, h_output, stream, d_input, d_output, sumtime = init(model_name)
    # 推理
    def inference(data_path):
        nonlocal context, h_input, h_output, stream, d_input, d_output, sumtime
        load_normalized_data(data_path, h_input,target_size,batch_size)
        t1 = time.time()
        # 将图片数据送到cuda显存中
        cuda.memcpy_htod_async(d_input, h_input, stream)
        # 模型预测
        context.execute_async(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
        # 将结果送回内存中
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        ## 异步等待结果
        stream.synchronize()
        # Return the host output.
        t2 = time.time()
        print(pid,"推理时间", t2 - t1)
        sumtime = sumtime + t2 - t1
        #print(h_output)
        #print(np.max(h_output), np.min(h_output))
        #h_output = np.reshape(h_output, (4,3,target_size[1], target_size[0]*3))
        return h_output
    start=time.time()
    #imgs_path = "data/video_test/decode/"
    imgs_path = "/home/ubuntu/csw/AccMPEG/VideoAnalysis/mi_dds_sr/Faster-RCNN/image_src/"
    imgs = os.listdir(imgs_path)
    pred = [[] for _ in range(3000)]
    for idx,img in enumerate(imgs):
        if idx==1500:break
        img_path = os.path.join(imgs_path, img)
        output = inference(data_path=img_path)
        #print(output.shape)
        copy_tmp = copy.deepcopy(output)
        #copy_tmp = np.argmax(copy_tmp, axis=0)
        #tmp = torch.from_numpy(copy_tmp).mul(1).clamp(0,255).round().div(1).detach().squeeze(0).byte().numpy()
        #res_img = tmp.transpose(1, 2, 0)
        #i = int(img_path.replace('.jpg','').replace('data/video_test/decode/frame-',''))
        i = int(img.split('.')[0])
        #if idx==16:break

        pred[i]=copy_tmp
        #cv2.imwrite('/home/ubuntu/csw/AccMPEG/VideoAnalysis/mi_dds_sr/Faster-RCNN/SR_part_image1/%010d.png'%i, res_img)
        #print("type  output:", type(copy_tmp), "output.shape:", copy_tmp.shape, "output:", copy_tmp, "max:", np.amax(copy_tmp), "\n")
    print("PID",str(pid),"用时:", time.time()-start)
    #mask_ = np.array(pred)
    #mask_ = mask.astype(np.int8)
    #print(np.amax(mask_), np.amin(mask_))
    #print(mask_)
    #mask = torch.from_numpy(mask_)
    #mask = mask.softmax(dim=1)[:,1:2,:,:]
    #print(mask_.shape)
    #torch.save(mask.to(torch.device('cpu')),"mask_AccSR_pixel_720_trt.pth")
    #with open('yanshou.txt', 'wb') as file:
    #    pickle.dump(mask_, file, protocol = 4)

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    for i in range(4):
        multiprocessing.Process(target=one_infer, args=(i,"EDSR.engine",(640,360),1)).start()
    #multiprocessing.Process(target=one_infer, args=(1,"/home/ubuntu/csw/AccMPEG/VideoAnalysis/mi_dds_sr/yolov5/yolov5s_1.engine",(1920,1088),1)).start()
    '''
    p2 = multiprocessing.Process(target=one_infer, args=(2,)) 
    p1.start()
    p2.start()
    '''
    
    