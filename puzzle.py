import torch
import numpy as np
from utils import resource_alloacte,process_masks
from skimage.measure import label, regionprops
import time
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
import cv2
max_area=0
w_max,h_max=0,0
all=0
sum_batch=[]

class Puzzle_3D:
    def __init__(self,height,width):
        self.height=height
        self.width=width
        self.layer=0
        self.c_area=[(0,0,width,height,0)]
        self.max_layer,_,_=resource_alloacte()
    def append(self,area):
        if area[0]>self.width and area[1]>self.height:
            self.layer+=1
            return 0,0,0,self.layer
        for idx,(x,y,w,h,l) in enumerate(self.c_area):
            if area[0]<=w and area[1]<=h:
                self.resize_area(area[0],area[1],idx)
                return 0,x,y,l
            elif area[0]<=h and area[1]<=w:
                self.resize_area(area[1],area[0],idx)
                return 1,x,y,l

        self.layer+=1

        self.c_area.append((0,0,self.width,self.height,self.layer))
        self.resize_area(area[0],area[1],-1)


        return 0,0,0,self.layer

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

class Puzzle_3D_com:
    def __init__(self,height,width):
        self.height=height
        self.width=width
        self.layer=0
        self.c_area=[(0,0,width,height,0)]
        self.max_layer,_,_=resource_alloacte()
    def append(self,area):
        if area[0]>self.width and area[1]>self.height:
            self.layer+=1
            return 0,0,0,self.layer
        for idx,(x,y,w,h,l) in enumerate(self.c_area):
            if area[0]<=w and area[1]<=h:
                self.resize_area(area[0],area[1],idx)
                return 0,x,y,l
            # elif area[0]<=h and area[1]<=w:
            #     self.resize_area(area[1],area[0],idx)
            #     return 1,x,y,l

        self.layer+=1

        self.c_area.append((0,0,self.width,self.height,self.layer))
        self.resize_area(area[0],area[1],-1)


        return 0,0,0,self.layer

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

class Puzzle_3D_G:
    def __init__(self,height,width):
        self.height=height
        self.width=width
        self.layer=0
        self.c_area=[(0,0,width,height,0)]
        self.max_layer,_,_=resource_alloacte()
    def append(self,area):
        if area[0]>self.width and area[1]>self.height:
            self.layer+=1
            return 0,0,0,self.layer
        for idx,(x,y,w,h,l) in enumerate(self.c_area):
            if area[0]<=w and area[1]<=h:
                self.resize_area(area[0],area[1],idx)
                return 0,x,y,l
            # elif area[0]<=h and area[1]<=w:
            #     self.resize_area(area[1],area[0],idx)
            #     return 1,x,y,l

        self.layer+=1

        self.c_area.append((0,0,self.width,self.height,self.layer))
        self.resize_area(area[0],area[1],-1)


        return 0,0,0,self.layer

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

def find_max_inner_rectangle(matrix):
    rows = len(matrix)
    cols = len(matrix[0])
    max_area = 0
    max_rect = (0, 0, 0, 0)

    left = [[0] * cols for _ in range(rows)]
    up = [[0] * cols for _ in range(rows)]
    # 初始化第一行和第一列
    for i in range(rows):
        left[i][0] = matrix[i][0]
    for j in range(cols):
        up[0][j] = matrix[0][j]
    # 更新 left 和 up 的值
    for i in range(rows):
        for j in range(cols):
            if matrix[i][j] == 1:
                left[i][j] = left[i][j-1] + 1
                up[i][j] = up[i-1][j] + 1
    # 计算最大面积和对应的矩形位置
    for i in range(rows):
        for j in range(cols):
            if matrix[i][j] == 1:
                width = left[i][j]
                height = up[i][j]
                min_width = width
                for k in range(height):
                    min_width = min(min_width, left[i-k][j])
                    area = (k + 1) * min_width
                    if area > max_area:
                        max_area = area
                        max_rect = (j - min_width + 1, i - k, j, i)
    return max_rect

def get_puzzle(masks,args,queue):
    masks=process_masks(masks)
    #global sum_batch
    p_3d=Puzzle_3D(360,640)
    bbox=[]
    t1=time.time()
    all_area=0

    block_size = 16
    for stream in range(masks.shape[0]):
        for frame_idx in range(masks.shape[1]):
            mask=masks[stream][frame_idx]
            labeled_img = label(mask, connectivity=2)
            for region in regionprops(labeled_img):
                minr, minc, maxr, maxc = region.bbox
                #maxr=min(22,maxr)
                if (maxc - minc)*(maxr - minr)>360*640/(8*block_size*block_size):
                    bbox.append([frame_idx, minc * block_size, minr * block_size, (maxc - minc) * block_size, int((maxr - minr) /2)*block_size,stream])
                    bbox.append([frame_idx, minc * block_size, int((minr + maxr) /2), (maxc - minc) * block_size, int((maxr - minr) /2)*block_size,stream])

                else:
                    bbox.append([frame_idx, minc * block_size, minr * block_size, (maxc - minc) * block_size, (maxr - minr) * block_size,stream])

    bbox = sorted(bbox, key=lambda x: x[3] * x[4], reverse=True)
    t3=0

    for idx in range(len(bbox)):
        #frame_idx,o_x,o_y,w,h,旋转,x,y,batch_idx
        flag, x, y, l=p_3d.append([bbox[idx][3],bbox[idx][4]])
        stream=bbox[idx][-1]
        # if  not flag:
        #     mask_bbox=[]
        #     mask = masks[stream][bbox[idx][0]]
        #     mask= mask[int(bbox[idx][2] / block_size):int(bbox[idx][4] / block_size + bbox[idx][2] / block_size), int(bbox[idx][1] / block_size):int(bbox[idx][3] / block_size + bbox[idx][1] / block_size)]
        #     one = np.ones_like(mask)*2
        #     mask = np.where(mask < 1, one, mask)
        #     zero = np.zeros_like(mask)
        #     mask = np.where(mask ==1, zero, mask)
        #     one = np.ones_like(mask)
        #     mask = np.where(mask == 2, one, mask)
        #
        #     minc, minr, maxc, maxr=find_max_inner_rectangle(np.array(mask,dtype=int))
        #
        #     # labeled_img = label(mask, connectivity=2)
        #     # if len(labeled_img)>0:
        #     #     for region in regionprops(labeled_img):
        #     #         minr, minc, maxr, maxc = region.bbox
        #     #         #print(region.bbox)
        #     p_3d.c_area.append((x + minc * block_size, y + minr * block_size, min(bbox[idx][3],(maxc - minc) * block_size), min(bbox[idx][4],(maxr - minr) * block_size), l))
        #     mask_bbox.append((minc * block_size, minr * block_size, min(bbox[idx][3],(maxc - minc) * block_size), min(bbox[idx][4],(maxr - minr) * block_size)))
        #
        #     bbox[idx]+=[flag,x,y,l,mask_bbox]
        # else:
        #     bbox[idx] += [flag, x, y, l,0]
        bbox[idx] += [flag, x, y, l, 0]
    #print(len(p_3d.c_area))
    # for x,y,w,h,l in p_3d.c_area:
    #     area_k+=w*h
    # print(area_k)
    # sum_batch.append(area_k/(360*640))
    logging.info("batchsize:%i", p_3d.layer)
    queue.put((bbox,p_3d.layer))
    return p_3d.layer

def puzzle_img(frame_list,bboxs,batch_size):
    global all
    batch_img=torch.zeros((batch_size,3,360,640)).cuda().float()
    for bbox in bboxs:
        frame_idx, o_x, o_y, w, h,stream, flag, x, y, batch_idx,_=bbox
        '''
        max_area=max(max_area,w*h)
        w_max=max(w_max,w)
        h_max=max(h_max,h)
        '''
        if batch_idx>=batch_size:
            continue
        oimg=frame_list[stream][frame_idx].clone()

        oimg=oimg[:,o_y:o_y+h,o_x:o_x+w]
        if flag:
            batch_img[batch_idx,:,y:y+w,x:x+h]=torch.transpose(oimg,1,2)
        else:
            #print(batch_idx)
            batch_img[batch_idx,:,y:y+h,x:x+w]=oimg
    # if batch_size>1:
    #     for i in range(batch_size):
    #         #print(batch_img[i].size())
    #         array = batch_img[i].cpu().numpy()
    #         array = array.transpose(1, 2, 0)
    #
    #         # Convert numpy array to BGR color format expected by OpenCV
    #         image = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    #
    #         # Save image to disk using OpenCV
    #         cv2.imwrite("puzzle_img/image%09d.png"% all, image)
    #
    #         all+=1
    #while 1:pass
    #print(max_area,w_max,h_max)
    return batch_img

def un_puzzle_img(frame_list,sr_img,bboxs,scale,batch_size):
    global all
    for bbox in bboxs:
        frame_idx, o_x, o_y, w, h, stream, flag, x, y, batch_idx,mask_bbox = bbox
        if batch_idx>=batch_size:
            continue
        oimg = sr_img[batch_idx].clone()
        #print(sr_img[batch_idx].shape)
        #oimg=oimg[:,:,0:50,0:10]
        if flag:
            oimg=oimg.squeeze()[:,  y*scale:(y + w)*scale,x*scale:(x + h)*scale]
            #print(o_y*scale,(o_y + w)*scale,o_x*scale,(o_x + h)*scale,y*scale,(y + h)*scale,x*scale,(x + w)*scale,oimg.shape)
            frame_list[stream][frame_idx][:, o_y*scale:(o_y + h)*scale,o_x*scale:(o_x + w)*scale] = torch.transpose(oimg, 1, 2)
        elif mask_bbox==0:
            oimg=oimg.squeeze()[:,  y*scale:(y + h)*scale,x*scale:(x + w)*scale]
            #print(oimg.shape)
            frame_list[stream][frame_idx][:, o_y*scale:(o_y + h)*scale,o_x*scale:(o_x + w)*scale] = oimg
        else:
            oimg = oimg.squeeze()[:, y * scale:(y + h) * scale, x * scale:(x + w) * scale].clone()
            for box in mask_bbox:
                m_x, m_y, m_w, m_h = box
                #print(box,bbox,oimg.shape)
                oimg[:, m_y * scale:(m_y + m_h) * scale,
                       m_x * scale:(m_x + m_w) * scale] = frame_list[stream][frame_idx][:,
                                                                    (o_y + m_y) * scale:(o_y + m_y + m_h) * scale,
                                                                    (o_x + m_x) * scale:(o_x + m_x + m_w) * scale].clone()

            # if stream==1 and frame_idx==23:
            #     for box in mask_bbox:
            #         m_x, m_y, m_w, m_h = box
            #         print(box)
            #     array = oimg.cpu().numpy()
            #     array = array.transpose(1, 2, 0)
            #
            #     # Convert numpy array to BGR color format expected by OpenCV
            #     image = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
            #
            #     # Save image to disk using OpenCV
            #
            #     cv2.imwrite("puzzle_img/aaa_4.png", image)
            #     while 1:pass
            frame_list[stream][frame_idx][:, o_y*scale:(o_y + h)*scale,o_x*scale:(o_x + w)*scale] = oimg


    # for stream in range(1):
    #     for frame_idx in range(10):
    #         array = frame_list[stream][frame_idx].cpu().numpy()
    #         array = array.transpose(1, 2, 0)
    #
    #         # Convert numpy array to BGR color format expected by OpenCV
    #         image = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    #
    #         # Save image to disk using OpenCV
    #         cv2.imwrite("puzzle_img/image%09d.png"% all, image)
            #all+=1
    #while 1:pass
    return frame_list