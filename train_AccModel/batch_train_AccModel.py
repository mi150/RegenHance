import os
import subprocess
from itertools import product

#app = "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
app = "EfficientDet"
architecture = "SSD"
model_name = f"AccModel"
filename = "accmpegmodel"

gt = "pickles/gt.pickle"

subprocess.run(
    [
        "python",
        "train_AccModel.py",
        #"--init",
        #f"maskgen_pths/{model_name}.pth",
        "--training_set",
        "Test",
        "--no_class_check",
        "-g",
        f"{gt}",
        "-p",
        f"maskgen_pths/{model_name}.pth",
        # "--init",
        # f"maskgen_pths/{model_name}.pth.best",
        "--tile_size",
        "16",
        "--batch_size",
        "4",
        "--log",
        f"train_{model_name}.log",
        "--maskgen_file",
        f"maskgen/SSD/{filename}.py",
        "--visualize",
        "False",
        "--visualize_step_size",
        "200",
        "--app",
        f"{app}",
        "--local_rank",
        "1",
        "--num_workers",
        "10",
        "--learning_rate",
        "1e-3",
        "--architecture",
        architecture,
        "--test_set",
        "RegenHance",
        "--confidence_threshold",
        "0.3",
        "--gt_confidence_threshold",
        "0.3",
    ]
)

