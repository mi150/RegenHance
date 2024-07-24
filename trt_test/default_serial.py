import ffmpegcv
import cv2
import time
import random
import torch
import torch.nn.functional as F
import imutils
import puzzle
from trt_infer import trt_infer
from sr_infer import SR_infer
from detect_infer import detect_infer
import argparse
import multiprocessing
from multiprocessing import Array,Lock,Value,Pool
import numpy as np
import logging
import pycuda.driver as cuda
from puzzle import sum_batch
from utils import global_block_list,resource_alloacte,process_masks,load_normalized_data
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
cfx=cuda.Device(0).make_context()

def parse_args():
    parser = argparse.ArgumentParser(description='Model prediction')

    # params of prediction
    parser.add_argument(
        "--config", dest="cfg", help="The config file.", default=None, type=str)

    parser.add_argument(
        '--scale',
        dest='scale',
        help='The scale of model for SR',
        type=int,
        default=3)
    parser.add_argument(
        '--thres',
        dest='thres',
        help='thres',
        default=5,
        type=float)
    parser.add_argument(
        '--max_batch_size',
        dest='max_batch_size',
        help='The max_batch_size of model for SR',
        type=int,
        default=4)
    parser.add_argument(
        '--sr_model_path',
        dest='sr_model_path',
        help='The path of model for SR',
        type=str,
        default=None)
    parser.add_argument(
        '--onnx_path',
        dest='onnx_path',
        help='The path of model for onnx',
        type=str,
        default=None)
    parser.add_argument(
        '--mask_model_path',
        dest='mask_model_path',
        help='The path of model for mask',
        type=str,
        default=None)
    parser.add_argument(
        '--detect_model_path',
        dest='detect_model_path',
        help='The path of model for detect',
        type=str,
        default=None)
    parser.add_argument(
        '--video_paths',
        dest='video_paths',
        help='The video to predict, which can be a path of image, or a file list containing image paths, or a directory including images',
        nargs='*', type=str,
        default=['input.mp4' for _ in range(6)])
    # set video height
    parser.add_argument(
        '--height',
        dest='height',
        help='video height',
        default='360',
        type=int)
    # set video width
    parser.add_argument(
        '--width',
        dest='width',
        help='video width',
        default='640',
        type=int)
    # set device
    parser.add_argument(
        '--device',
        dest='device',
        help='Device place to be set, which can be GPU, XPU, NPU, CPU',
        default='CUDA',
        type=str)
    # save results
    parser.add_argument(
        '--save',
        dest='save',
        help='Save SR results',
        default=False,
        type=bool)
    # save GOP
    parser.add_argument(
        '--gop',
        dest='gop',
        help='video gop',
        default=30,
        type=int)
    return parser.parse_args()

class Puzzle_3D:
    def __init__(self,width,height):
        self.height=height
        self.width=width
        self.layer=0
        self.c_area=[(0,0,width,height,0)]
    def append(self,area):
        if area[0]>self.width or area[1]>self.height:
            return -1
        for idx,(x,y,w,h,l) in enumerate(self.c_area):
            if area[0]<=w and area[1]<=h:
                self.resize_area(area[0],area[1],idx)
                return (0,x,y,l)
            elif area[0]<=h and area[1]<=w:
                self.resize_area(area[1],area[0],idx)
                return (1,x,y,l)
        self.layer+=1
        self.c_area.append((0,0,self.height,self.width,self.layer))
        self.resize_area(area[0],area[1],-1)
        return (0,0,0,self.layer)

    def reset(self):
        self.layer=0
        self.c_area=[(0,0,self.width,self.height,0)]

    def resize_area(self,a_w,a_h,idx):
        x,y,w,h,l=self.c_area[idx]
        self.c_area.pop(idx)
        if (w-a_w)*h>=(h-a_h)*w:
            self.c_area.append((x + a_w, y, w - a_w, h, l))
            self.c_area.append((x, y + a_h, a_w, h-a_h, l))
        else:
            self.c_area.append((x, y + a_h, w, h-a_h, l))
            self.c_area.append((x + a_w, y, w - a_w, a_h, l))


def maxminnorm(array):
    _max=max(array)
    _min=min(array)
    t=[]
    for i in array:
        t.append((i-_min)/(_max-_min))
    return t
def get_frame_feature(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #return gray
    blur = cv2.GaussianBlur(gray, (5, 5),0)
    #return blur
    #边缘检测
    # gray_lap = cv2.Laplacian(blur, cv2.CV_16S, ksize=3)
    # dst = cv2.convertScaleAbs(gray_lap)
    edge = cv2.Canny(blur, 101,  255)

    return edge
def cal_frame_diff(frame, prev_frame):
    # edge=np.abs(frame-prev_frame)
    # return np.sum(edge)
    #total_pixels = frame.shape[0] * frame.shape[1]

    frame_delta = cv2.absdiff(frame, prev_frame)
    thresh = cv2.threshold(frame_delta, 101, 255,
                           cv2.THRESH_BINARY)[1]
    ###
    #膨胀
    thresh = cv2.dilate(thresh, None)
    #寻找图像中的轮廓
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    if not contours:
        return 0.000000001
    #返回最大面积的轮廓，即两帧之间变化的最大面积
    return max([1/(1+cv2.contourArea(c))  for c in contours])

def warmup(args,cfx):
    sr=SR_infer(args,cfx)
    mask=trt_infer(args,cfx,"CPU")
    detect=detect_infer(args,cfx)
    #seg=seg_infer(args,cfx)
    #while 1:pass

    return mask,sr,detect

def get_feature(frame_list):
    feature_list=[0]
    prev=get_frame_feature(frame_list[0])
    for frame in frame_list[1:]:
        feat=get_frame_feature(frame)
        feature_list.append(cal_frame_diff(prev,feat)+feature_list[-1])

    return maxminnorm(feature_list)

def get_frame_index(diff_list,num):
    thres=float(1/num)
    start=0
    index=[]
    for end,diff in enumerate(diff_list):
        if diff>thres:
            index.append(random.choices(range(start,end-1),diff_list[start:end-1])[0])
            start=end
            thres= 1 if (1-thres-float(1/num))<0.001 else thres+float(1/num)
        pass
    index.append(random.choices(range(start, len(diff_list)), diff_list[start:len(diff_list)])[0])
    return index

def get_select_frame(cap):

    fid=0
    frame_list=list()
    while True and fid<args.gop:

        ret, frame = cap.read()
        fid+=1

        frame_list.append(load_normalized_data(frame))
        if frame is None:
            return np.array(frame_list),False

    return np.array(frame_list),ret

def preprocess_single_stream(i):
    # video_path = video_paths[i]
    # cap = ffmpegcv.VideoCapture(video_path)
    t1 = time.time()
    cap=cap_list[i]
    #select chunk frame list
    t1 = time.time()
    masks=[]
    frame_list,ret=get_select_frame(cap)
    ret_list[i]=ret
    # if not ret:
    #     break
    #get feature
    ##feature_list=get_feature(frame_list)
    #select frame
    #num_list=get_frame_index(feature_list,num)
    num_list=range(args.gop)
    t2 = time.time()
    #logging.info("preprocess stream %i time: %f",i,t2-t1)

    #global_frame_list[i * 30 * 3 * 360 * 640:(i + 1) * 30 * 3 * 360 * 640]=np.array(frame_list).ravel()
    return np.squeeze(frame_list,axis=1)
    #global_masks[i*30*22*40:(i+1)*30*22*40]=np.array(masks).ravel()



def genearate_SR_region(frame_list,masks,args):
    t3 = time.time()
    # puzzle
    bboxs, batch_size = puzzle.get_puzzle(masks, args)  # 获取拼图后的位置
    logging.info("batch_size: %i", batch_size)
    batch_img = puzzle.puzzle_img(frame_list, bboxs, batch_size)  # 拼到tensor上
    t4 = time.time()
    #logging.info("puzzle time: %f", t4 - t3)
    return batch_img,bboxs

def process_SR_Infer(args,sr,detect,batch_img,frame_list,bboxs):

    t4 = time.time()
    # SR
    sr_img = sr.inference(batch_img)  # sr
    t5 = time.time()
    logging.info("SR time: %f", t5 - t4)
    #process puzzle image

    frame_list=frame_list.view(frame_list.shape[0]*frame_list.shape[1],3,frame_list.shape[3],frame_list.shape[4])
    frame_list=F.interpolate(frame_list,size=(frame_list.shape[2]*args.scale, frame_list.shape[3]*args.scale), mode='bilinear', align_corners=False)
    frame_list=frame_list.view(len(args.video_paths),args.gop,3,frame_list.shape[2], frame_list.shape[3])
    frame_list = puzzle.un_puzzle_img(frame_list, sr_img, bboxs, args.scale)  # 将sr后的区域拼回原图像

    t6 = time.time()
    # infer
    detect_result = detect.inference(frame_list)
    t7 = time.time()
    logging.info("infer time: %f", t7 - t6)
    torch.cuda.empty_cache()

def GPU_process(frame_list,batch_img,bboxs,args):
    t1 = time.time()
    cfx = cuda.Device(0).make_context()

    maskgen, sr, detect = warmup(args,cfx)
    t2=time.time()
    #print(t2-t1)
    # sr=share_dict['sr']
    # detect=share_dict['detect']
    # generate SR region and puzzle


    # exec SR and Infer

    process_SR_Infer(args,sr,detect,batch_img, frame_list, bboxs)

    #cfx.pop()


if __name__=="__main__":
    #init pipeline
    args=parse_args()
    video_paths=args.video_paths
    stream_num=len(video_paths)
    num=3
    p=None
    multiprocessing.set_start_method('spawn')
    batch_size,_,_=resource_alloacte()
    # lock=Lock()
    # manager=multiprocessing.Manager()
    # share_cuda_handle=manager.Value(cfx.handle,"cuda_handle")
    #global_block_list=global_block_list()
    width, height=args.width,args.height
    #maskgen=trt_infer(args,[],"CPU")
    maskgen,sr,detect=warmup(args,cfx)
    # manager=multiprocessing.Manager()
    # share_dict=manager.dict()
    # share_dict['sr']=sr
    # share_dict['detect']=detect
    #queue = multiprocessing.Queue()

    #init frames,masks
    cap_list=[]
    ret_list=[]
    for video_path in video_paths:
        cap = ffmpegcv.VideoCapture(video_path)
        cap_list.append(cap)
        ret=Value('i',1)
        ret_list.append(ret)
    #frame_array=Array('f',[0]*(stream_num*30*3*360*640))
    frame_list=np.zeros((stream_num,args.gop,3,360,640),dtype='float32')
    #mask_array=Array('f',[0]*(stream_num*30*22*40))

    #get decode frame to frame_list and get maskgen
    start = time.time()
    for i,video_path in enumerate(video_paths):
        if ret_list[i]:
            frame_list[i]=preprocess_single_stream(i)
            while 1:pass

    mask_list = np.zeros((stream_num,args.gop,22,40))
    for stream in range(stream_num):
        for index in range(args.gop):
            mask_list[stream][index]=maskgen.inference(np.expand_dims(frame_list[stream][index],axis=0))

    frame_array=torch.from_numpy(frame_list).to(
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    # get sr_block_list and global_block_list
    mask_list = process_masks(np.array(mask_list))
    batch_img, bboxs = genearate_SR_region(frame_array, mask_list, args)
    end = time.time()
    logging.info("get sr_block_list time: %f", end - start)

    while sum(ret_list)!=0:

        if p!=None:
            p.join()
        t5=time.time()
        #GPU-cuda-para
        batch_size=batch_img.shape[0]

        idx=0
        sr_img = [torch.empty(1, 3, 360*3, 640*3).cuda().int().detach() for _ in range(batch_size)]
        t1=time.time()
        while idx<batch_size:
            img=batch_img[idx].unsqueeze(0)
            # 模型预测
            sr.cfx.push()
            sr.context.execute_async(bindings=[img.data_ptr(), sr_img[idx].data_ptr()], stream_handle=sr.stream.handle)
            sr.cfx.pop()
            idx += 1
        #CPU preprocess

        frame_list = np.zeros((stream_num, args.gop, 3, 360, 640), dtype='float32')
        for i, video_path in enumerate(video_paths):
            if ret_list[i]:
                frame_list[i] = preprocess_single_stream(i)
        t6 = time.time()
        logging.info("decode time: %f", t6 - t5)
        mask_list = np.zeros((stream_num, args.gop, 22, 40))
        for stream in range(stream_num):
            for index in range(0,args.gop,5):
                mask_list[stream][index] = maskgen.inference(np.expand_dims(frame_list[stream][index], axis=0))
                mask_list[stream][index+1]=mask_list[stream][index]
                mask_list[stream][index + 2] = mask_list[stream][index]


        # get sr_block_list and global_block_list
        mask_list = process_masks(np.array(mask_list))
        t6 = time.time()
        logging.info("para time: %f", t6 - t5)
        sr.stream.synchronize()


        frame_array = frame_array.view(frame_array.shape[0] * frame_array.shape[1], 3, frame_array.shape[3],
                                     frame_array.shape[4])
        frame_array = F.interpolate(frame_array,
                                   size=(frame_array.shape[2] * args.scale, frame_array.shape[3] * args.scale),
                                   mode='bilinear', align_corners=False)
        frame_array = frame_array.view(len(args.video_paths), args.gop, 3, frame_array.shape[2], frame_array.shape[3])
        frame_array = puzzle.un_puzzle_img(frame_array, sr_img, bboxs, args.scale)  # 将sr后的区域拼回原图像

        t6 = time.time()
        logging.info("SR time: %f", t6 - t5)
        # infer
        detect_result = detect.inference(frame_array)
        t7 = time.time()
        logging.info("infer time: %f", t7 - t6)
        torch.cuda.empty_cache()
        start = time.time()
        frame_array = torch.from_numpy(frame_list).to(
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        batch_img, bboxs = genearate_SR_region(frame_array, mask_list, args)
        #p=multiprocessing.Process(target=GPU_process,args=(frame_list,batch_img, bboxs,args))
        #p.start()
        #generate SR region and puzzle
        # batch_img,bboxs=genearate_SR_region(frame_list, mask_list)
        #
        # #exec SR and Infer
        # process_SR_Infer(batch_img,frame_list,bboxs)

        #start = time.time()

        # batch_img_ptr=[]

        end = time.time()
        logging.info("cpu to gpu trans time: %f", end - start)
    #cfx.pop()
    # cpu,gpu=[],[]
    # while gpu.start:
    #     cpu,gpu