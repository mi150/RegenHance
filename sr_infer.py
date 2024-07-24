import tensorrt as trt
import torch
import pickle
import numpy as np
import pycuda.driver as cuda
import time
import cv2
import os
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import copy
import warnings

# 忽略TensorRT的警告
warnings.filterwarnings("ignore", category=Warning)
all=0
# 初始化(创建引擎，为输入输出开辟&分配显存/内存.)
class SR_infer():

    def __init__(self,args,cfx):
        model_path =args.sr_model_path
        self.cfx= cfx
        # 加载runtime，记录log
        self.runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
        # 反序列化模型
        self.engine = self.runtime.deserialize_cuda_engine(open(model_path, "rb").read())
        # 1. Allocate some host and device buffers for inputs and outputs:
        #self.max_batchsize=args.max_batchsize
        #context = self.engine.create_execution_context()
        #context.set_binding_shape(0, (1, 3, 293, 503))
        self.context = self.engine.create_execution_context()
        #self.context.set_binding_shape(0, (batch_size, 3, 326, 614))
        #self.h_input = cuda.pagelocked_empty(abs(trt.volume(self.engine.get_binding_shape(0))), dtype=trt.nptype(trt.float32))
        #self.h_output = cuda.pagelocked_empty(abs(trt.volume(self.engine.get_binding_shape(1))), dtype=trt.nptype(trt.int32))

        # Allocate device memory for inputs and outputs.
        #self.d_input = cuda.mem_alloc(self.h_input.nbytes)
        #self.d_output = cuda.mem_alloc(self.h_output.nbytes)
        # Create a stream in which to copy inputs/outputs and run inference.
        self.stream = cuda.Stream()
        self.context.set_binding_shape(0, (1, 3, 360, 640))
        # 推理上下文
        #self.context = self.engine.create_execution_context()
        

    # 推理
    def inference(self,imgs):
        global all
        batch_size=imgs.shape[0]
        idx=0
        output_datas = [torch.empty(1, 3, 360*3, 640*3).cuda().float().detach() for _ in range(int(batch_size))]

        while idx<batch_size:
            img=imgs[idx]
            t1 = time.time()
            #input_data = img.data_ptr()
            # 模型预测
            self.cfx.push()
            self.context.execute_async(bindings=[int(img.data_ptr()), int(output_datas[int(idx)].data_ptr())], stream_handle=self.stream.handle)
            self.cfx.pop()

            self.stream.synchronize()
            # array = output_datas[idx][0].cpu().numpy()
            # array = array.transpose(1, 2, 0)
            #
            # # Convert numpy array to BGR color format expected by OpenCV
            # image = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
            #
            # # Save image to disk using OpenCV
            #
            # cv2.imwrite("puzzle_img/image%09d.png" % all, image)
            # all+=1
            t2=time.time()
            print(t2-t1)
            idx += 1

        return output_datas


    # 加载数据并将其喂入提供的pagelocked_buffer中.
    def load_normalized_data(self,img, pagelocked_buffer, target_size=(640, 360)):
        img = cv2.resize(img, target_size, cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(np.array([img], dtype="float32"), (0, 3, 1, 2))
        np.copyto(pagelocked_buffer, img.ravel())


# import subprocess
# import numpy as np
# import cv2
#
# def read_frame_data(width, height, ffmpeg_process):
#     frame_size = width * height * 3
#     frame_data = ffmpeg_process.stdout.read(frame_size)
#     frame_array = np.frombuffer(frame_data, np.uint8)
#     return frame_array.reshape((height, width, 3))
#
# def read_log_data(ffmpeg_process):
#     log_data = ffmpeg_process.stderr.readline().decode("utf-8")
#     return log_data
#
# def main():
#     input_video = "D:\\VASRL\\input.mp4"
#     width = 640
#     height = 360
#
#     ffmpeg_command = [
#         "ffmpeg",
#         "-i", input_video,
#         "-vf", f"scale={width}:{height}",
#         "-pix_fmt", "bgr24",
#         "-an",  # 禁用音频
#         "-f", "rawvideo",
#         "-loglevel", "info",
#         "-",
#     ]
#
#     ffmpeg_process = subprocess.Popen(
#         ffmpeg_command,
#         stdout=subprocess.PIPE,
#         stderr=subprocess.PIPE,
#         bufsize=10**8,
#     )
#     i=0
#     while True:
#         # log_data = read_log_data(ffmpeg_process)
#         # print(log_data.strip())
#
#         frame_array = read_frame_data(width, height, ffmpeg_process)
#         print(i)
#         i+=1
#         if frame_array.size == 0:
#             break
#
#         # cv2.imshow("Video", frame_array)
#         # if cv2.waitKey(1) & 0xFF == ord("q"):
#         #     break
#
#     ffmpeg_process.terminate()
#     cv2.destroyAllWindows()
#
# if __name__ == "__main__":
#     main()