import cv2 as cv
import numpy as np
import random
mb_num=[]
class global_block_list:
    def __init__(self,height=360,width=640):
        self.score_list=[]
        self.height=height
        self.width=width
        self.puzzle_limit=0.2

    def __add__(self, other):
        self.puzzle_limit+=other.puzzle_limit
        del other

    def set_puzzle_limit(self,limit):
        self.puzzle_limit=limit

    def append(self,block):
        #block(score,stream_index,frame_index,y,x)
        self.score_list.append(block)

    def sort(self):
        random.shuffle(self.score_list)
        self.score_list = sorted(self.score_list, key=lambda x: x[0],reverse=True)

    def clean(self):
        self.score_list=[]

    def get_blocks(self,batch_size):
        require_block_nums=int((self.height*self.width*batch_size*self.puzzle_limit)/256)
        #print(self.score_list[:10])
        return self.score_list[:require_block_nums]
#resource management
def resource_alloacte():
    batchsize, height, width=20,360,640
    return batchsize,height,width

def load_normalized_data(img, target_size=(640, 360)):
    img = cv.resize(img, target_size, cv.INTER_CUBIC)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = np.transpose(np.array([img], dtype="float32"), (0, 3, 1, 2))
    return img

#(stream_num,30,22,40)
def process_masks(masks):
    global mb_num
    g_b_list=global_block_list()
    for stream in range(masks.shape[0]):
        for frame in range(masks.shape[1]):
            for y in range(masks.shape[2]):
                for x in range(masks.shape[3]):
                    g_b_list.append((masks[stream][frame][y][x],stream,frame,y,x))
    g_b_list.sort()
    sr_block_list=g_b_list.get_blocks(resource_alloacte()[0])
    #mb_num=[0 for _ in range(masks.shape[0])]
    for block in sr_block_list:
        _,stream,index,y,x=block
        masks[stream][index][y][x]=1
    #     mb_num[stream]+=1
    # print(mb_num)
    return masks

def save_SR_results(sr_results,stm_index):
    for idx,result in enumerate(sr_results):
        cv.imwrite(f"results/sr_{stm_index}_{idx}.png",result)

