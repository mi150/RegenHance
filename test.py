# import pycuda.autoinit
# import pycuda.driver as cuda
# import torch
# import time
# torch.cuda.empty_cache()
# time.sleep(5)
# (free, total) = cuda.mem_get_info()
# print("Global memory occupancy:%f%% free" % (free * 100 / total))
#
# for devicenum in range(cuda.Device.count()):
#     device = cuda.Device(devicenum)
#     attrs = device.get_attributes()
#
#     # Beyond this point is just pretty printing
#     print("\n===Attributes for device %d" % devicenum)
#     # for (key, value) in attrs.iteritems():
#     #     print("%s:%s" % (str(key), str(value)))
import multiprocessing as mp
import random
from puzzle import get_puzzle,Puzzle_3D,Puzzle_3D_G
import numpy as np
import time
import torch
import asyncio
from sr_infer import SR_infer
from torchvision.ops import box_convert,nms
from trt_test.dds_utils import evaluate, read_results_dict, merge_boxes_in_results,cal_miou
import pickle
ff=np.zeros((3,360,640,120),dtype=int)

def test(i,queue):
    t1 = time.time()
    # ff=np.zeros((3,360,640,10),dtype=int)
    q=np.ones_like((1,22,40,30),dtype=int)
    q=q*2
    # ff=list(ff.ravel())
    #time.sleep(2)
    t2 = time.time()
    print(q)
    #return 1,np.zeros((3,360,640,120),dtype=int)
    # print(frame)
import pycuda.driver as cuda
# [0.735, 0.855, 0.88, 0.871, 0.846, 1.0, 0.968, 0.75, 0.933, 0.971, 0.93, 0.922, 0.945, 0.959, 0.907]
# [0.932, 0.947, 0.943, 0.943, 0.971, 0.982, 0.787, 0.893, 0.89, 0.889, 0.742, 0.793, 0.814, 0.718, 0.852]
# [0.95, 0.952, 0.94, 0.959, 0.972, 0.956, 0.978, 0.993, 0.979, 0.977, 0.974, 0.972, 0.981, 0.976, 0.969]
# [0.923, 0.926, 0.961, 0.924, 0.924, 0.945, 0.95, 0.948, 0.962, 0.974, 0.974, 0.935, 0.89, 0.904, 0.892]
# [0.903, 0.941, 0.935, 0.934, 0.938, 0.942, 0.957, 0.963, 0.957, 0.956, 0.95, 0.92, 0.924, 0.96, 0.968]
# [0.985, 0.95, 0.915, 0.955, 0.981, 0.976, 0.979, 0.963, 0.989, 0.975, 0.982, 0.995, 0.987, 0.99, 0.947]
# [6914, 9421, 9480, 9222, 8608, 10355]
# [6752, 9461, 9449, 9264, 8630, 10444]
# [6562, 9298, 9798, 9528, 8934, 9880]
# [6929, 9202, 9879, 8164, 8245, 11581]
# [5448, 9607, 9990, 9184, 9264, 10507]
# [6351, 10950, 9368, 9169, 8895, 9267]
# [4223, 6163, 13550, 10088, 10249, 9727]
# [3711, 7201, 13550, 9864, 10524, 9150]
# [3542, 6286, 14021, 10143, 10461, 9547]
# [3719, 7310, 11833, 10443, 10633, 10062]
# [4045, 6894, 11953, 10082, 10427, 10599]
# [8186, 4683, 11689, 9922, 8506, 11014]
# [9629, 5840, 13547, 7251, 7598, 10135]
# [7946, 5879, 12227, 7352, 8967, 11629]
# [7230, 6240, 12685, 6686, 10230, 10929]
# [7088, 6039, 12876, 7389, 10086, 10522]
# [9850, 9881, 9775, 9099, 10786, 4609]
# [9973, 10379, 10066, 9678, 10440, 3464]
# [9904, 10599, 8932, 8987, 12169, 3409]
# [9915, 10307, 9573, 9580, 10844, 3781]
# [11348, 9834, 9749, 9463, 9844, 3762]
# [6235, 13660, 10091, 10337, 9856, 3821]
# [7353, 13659, 9908, 10696, 9318, 3066]
# [6592, 14138, 10305, 10660, 9688, 2617]
# [7505, 12053, 10616, 10673, 10112, 3041]
# [7086, 12117, 10250, 10593, 10808, 3146]
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
                        max_rect = (j - min_width + 1, i - k, j+1, i+1)
    return max_rect

if __name__=="__main__":




    # # 示例矩阵
    # matrix = [
    #     [1, 1, 1, 0, 1],
    #     [0, 1, 1, 1, 0],
    #     [1, 1, 1, 1, 1],
    #     [1, 0, 1, 1, 1],
    #     [1, 0, 1, 1, 0]
    # ]
    # t1=time.time()
    # max_inner_rectangle = find_max_inner_rectangle(matrix)
    # print("最大内接矩形的位置和尺寸：", max_inner_rectangle)
    # t2 = time.time()
    # print(t2-t1)
    #detect

    # for i in range(2):
    #     gt = read_results_dict('/home/dodo/trt_test/profile/results/result_gt_%d'%0)
    #
    #     result = read_results_dict('/home/dodo/trt_test/profile/results/result_js_2_%d'%i)
    #     #result=read_results_dict('video_test_360p')
    #     # gt = merge_boxes_in_results(gt, 0.25, 0.3).regions_dict
    #     # result = merge_boxes_in_results(result, 0.25, 0.3).regions_dict
    #     f1_list=[]
    #     tp, fp, fn, _, _, _, f1 = evaluate(
    #         0, 2999, result, gt,
    #         0.5, 0.5, 0.4, 0.4)
    #     # for j in range(15):
    #     #     tp, fp, fn, _, _, _, f1 = evaluate(
    #     #         450*i+j*30,450*i+j*30+30, result, gt,
    #     #         0.5, 0.5, 0.4, 0.4)
    #     #     f1_list.append(f1)
    #     print(f1)
    #gt = merge_boxes_in_results(gt, 0.25, 0.3).regions_dict

    #Seg#差值0.81
    # avg_miou=[]
    for j in range(1):
        avg_miou = []
        for i in range(1100):
            file= open('seg_results/result_str_%d%d.txt'%(j,i), 'rb')
            #file=open('seg_results/result_str_gt_0%d.txt'%i, 'rb')
            pred=pickle.load(file)
            save_pred = pred
            pred[np.where(pred <= 10)] = 0
            pred[np.where((pred > 10) & (pred <= 12))] = 1
            pred[np.where(pred > 12)] = 2
            gt_file= open('seg_results/result_str_gt_0%d.txt'%i, 'rb')
            truth=pickle.load(gt_file)
            truth = truth
            truth[np.where(truth <= 10)] = 0
            truth[np.where((truth > 10) & (truth <= 12))] = 1
            truth[np.where(truth > 12)] = 2
            # print(list(pred),list(truth))
            # while 1:pass
            miou=cal_miou(truth, pred, [0,1,2], n_classes=3)
            #print(miou)
            avg_miou.append(miou)
        print(np.mean(avg_miou))

    # test(0,0)
    # output=torch.rand(1, 1285200,85).cuda()
    # nms_threshold = 0.5
    # confidence_threshold = 0.5
    # boxes = output[..., :4].squeeze().contiguous()
    # scores = output[..., 4].squeeze().contiguous()
    # boxes = box_convert(boxes, 'cxcywh', 'xyxy')
    #
    #
    # keep_indices = nms(boxes, scores, nms_threshold)
    # print(boxes.shape)
    # frame=mp.Array('i',3*360*640)
    # queue=mp.Queue()
    #
    # #with mp.Pool() as pool:
    # results = []
    # p=[]
    # t1=time.time()
    # for i in range(6):
    #     p.append(mp.Process(target=test,args=(0,queue)))
    #     p[-1].start()
    # #time.sleep(10)
    # for i in range(6):
    #     t2 = time.time()
    #     print(t2 - t1)
    #     result = queue.get()
    #     print(result.shape)
        #p[i].join()

    # for i in range(4):
    #     result=pool.apply_async(test,(i,queue))
    #     results.append(result)
    #
    # for result in results:
    #     res=result.get()


    #ll=np.frombuffer(frame.get_obj(),dtype='i')

    #puzzle
    # ours=[]
    # greedy=[]
    # gt = read_results_dict('/home/dodo/trt_test/profile/results/result_gt_0')
    # for i in range(1000):
    #     bboxs=[]
    #     bbox=[]
    #     random_nums=random.sample(range(2000),100)
    #     for j in random_nums:
    #         for box in gt[j]:
    #             bboxs.append(box)
    #     p_3d=Puzzle_3D(360,640)
    #     p_3d_g = Puzzle_3D_G(360, 640)
    #     area=0
    #     for box in bboxs:
    #         w, h= box.w,box.h
    #         w,h=int(w*360),int(h*640)
    #         bbox.append((w,h))
    #         area+=w*h
    #         #p_3d.append([w,h])
    #         p_3d_g.append([w, h])
    #     bboxs = sorted(bbox, key=lambda x: x[0] * x[1], reverse=True)
    #     for box in bboxs:
    #         w, h= box[0],box[1]
    #         if w*h>360*100:
    #             p_3d.append([w/2, h])
    #             p_3d.append([w/2, h])
    #             continue
    #         #w,h=int(w*360),int(h*640)
    #         p_3d.append([w,h])
    #     #if p_3d.layer!=p_3d_g.layer:
    #     ours.append(area/((p_3d.layer+1)*360*640))
    #     greedy.append(area/((1+p_3d_g.layer)*360*640))
    #     print(i)
    #     #while 1:pass
    # print('average:')
    # print(sum(ours)/len(ours))
    # print(sum(greedy)/len(greedy))
    # print('95%:')
    # ours.sort()
    # greedy.sort()
    # print(ours[int(0.05*len(ours))])
    # print(greedy[int(0.05 * len(greedy))])
    # print('90%:')
    # print(ours[int(0.1 * len(ours))])
    # print(greedy[int(0.1 * len(greedy))])


