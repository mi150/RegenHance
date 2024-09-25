import os
import subprocess
from itertools import product
from config import settings

import yaml

x264_dir = settings.x264_dir

conv_list = [1]
bound_list = [0.2]
base_list = [40]
tile = 16

v_list = ["input"]

# efficientdet
# stats = "frozen_stats_MLSys/stats_QP30_thresh4_segment_EfficientDet"
conf_thresh = 0.3
gt_conf_thresh = 0.3
app_name = "EfficientDet"
model_app = "FPN"

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

    # output = f'{v}_compressed_ground_truth_2%_tile_16.mp4'
    # visdrone/videos/vis_169_blackgen_bound_0.2_qp_30_conv_5_app_FPN.mp4
    # output = f"{v}_blackgen_bound_{bound}_qp_30_conv_{conv}_app_FPN.mp4"

    output = f"{v}_app_{model_app}.mp4"

    # examine_output = (
    #     f"{v}_blackgen_dual_SSD_bound_{bound}_conv_{conv}_app_FPN.mp4"
    # )

    # os.system(f"rm -r {examine_output}*")

    #if not os.path.exists(output):
    if True:

        os.system(
            f"python compress_blackgen_roi_test.py -i {v}.mp4 "
            f" {v}.mp4 -s {v} -o {output} --tile_size {tile}  -p maskgen_pths/{model_name}.pth.best"
            f" --conv_size {conv} "
            f" -g {v}.mp4 --bound {bound} --smooth_frames 1 --app {app_name} "
            f"--maskgen_file maskgen/SSD/{filename}.py --visualize_step_size {visualize_step_size}"
        )