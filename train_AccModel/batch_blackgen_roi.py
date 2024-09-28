import os
import subprocess
from itertools import product

import yaml

conv_list = [1]
bound_list = [0.2]
base_list = [40]
tile = 16

v_list = ["input"]

# efficientdet
conf_thresh = 0.3
gt_conf_thresh = 0.3
app_name = "EfficientDet"

model_name = "AccModel"

# model_name = "pretrainedkeypointmodel"
# model_app = "fcn"


visualize_step_size = 10000
# accs = [filter([fmt % i, "newSSDwconf", "bound_0.2", "lq_40", "conv_1"]) for i in ids]

import glob

# app_name = "Segmentation/fcn_resnet50"
# app_name = "EfficientDet"
filename = "SSD/accmpegmodel"

for conv, bound, base, v in product(conv_list, bound_list, base_list, v_list):

    print(v, conv, bound, base)

    output = f"{v}_app_{app_name}.mp4"

    #if not os.path.exists(output):
    if True:

        os.system(
            f"python compress_blackgen_roi.py -i {v}.mp4 "
            f" {v}.mp4 -s {v} -o {output} --tile_size {tile}  -p maskgen_pths/{model_name}.pth.best"
            f" --conv_size {conv} "
            f" -g {v}.mp4 --bound {bound} --smooth_frames 1 --app {app_name} "
            f"--maskgen_file maskgen/{filename}.py --visualize_step_size {visualize_step_size}"
        )
