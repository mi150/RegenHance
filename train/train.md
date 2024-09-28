[🇺🇸](/train.md "English") [🇨🇳](/train-cn.md "Simplified Chinese")

# Train a macroblock-based region importance predictor

This README file will guide you step by step to implement training a macroblock-level region importance predictor on your own dataset. The training code has not been maintained for a long time and some bugs may be encountered. If you encounter any problems at following steps, you are welcome to leave a message and ask questions in **[Issues](https://github.com/mi150/RegenHance)**. ❤️

## 1. Set up the environment

First, git clone our repo to your working directory $DIR. Then, install the corresponding conda environment through the `conda_env.yml` file. If there is an error when installing some libraries using pip, you can try changing the source (such as Tsinghua source) and then download again through `train_AccModel/requirements.txt`.

```bash
git clone https://github.com/mi150/RegenHance.git
conda env create -f conda_env.yml
```

Then activate the installed environment: 

```bash
conda activate regenhance
```

After configuring the environment, please install `pytorch` and `torchvision` through **`pip`** (you can refer to this link: [Pytorch](https://pytorch.org/get-started/locally/). We use pytorch=1.10.1 and the CUDA environment is 11.1). You also need to install `detectron2` (you can refer to [detectron2](https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md), and pay attention to matching the correct pytorch and CUDA versions). 

The above process will automatically install an old version of `ffmpeg`. It needs to be replaced with a new version. Please download a static version of ffmpeg from [FFmpeg](https://johnvansickle.com/ffmpeg/) and use it to replace the original ffmpeg (you can find the original ffmpeg through the `which ffmpeg` command. The ffmpeg version we use is 5.0.1). 

Since super-resolution is required when preparing data later, you can choose an appropriate super-resolution model. We have selected the `EDSR` model as the super-resolution model. For specific environment configuration, you can refer to their official github of [EDSR](https://github.com/sanghyun-son/EDSR-PyTorch).

## 2. Prepare data

Return to your working directory $DIR and prepare the video data you wish to train on. We provide an `input.mp4` as a reference. Taking vehicle and pedestrian detection as an example, reproduce the training process. 

Run `extract.py` to extract images in png format from the `input.mp4` video. These images will serve as the original input of the video at a low resolution of 360p. 

During the training process, in order to calculate regional importance, we need blurred input and clear input.  The blurred input is the above low-resolution image. The clear input is got through super-resolution. You need to prepare a trained super-resolution model in advance. We use the [EDSR](https://github.com/sanghyun-son/EDSR-PyTorch) super-resolution model. Other models are also possible. Just configure different models according to the multiple of resolution you hope to improve. 

Since full-frame super-resolution has a large computational cost, we recommend completing the super-resolution of low-resolution images in advance before training the macroblock-based region importance predictor and saving them in a specific directory.

After completing this step, organize your dataset into a structure similar to the following:

```tex
YOUR_dataset
    |
    |--or_images          # Store all original low-resolution images.
    |  |--0000000000.png
    |  |--0000000001.png
    |  |--...
    |
    |--sr_images          # Store all super-resolution clear images.
    |  |--0000000000.png
    |  |--0000000001.png
    |  |--...
```

## 3. Train model

After making the above preparations, enter the directory of $DIR/train_AccModel, modify the dataset path in the file `train_AccModel.py`, and then run `batch_train_AccModel.py` to start training the macroblock-based region importance predictor. (Currently, the code defaults to training when super-resolving 360p to 1080p, that is, a magnification factor of 3. If it is other sizes and magnification factors, please modify the size in the following files).

```tex
1. maskgen/SSD/accmpegmodel.py: In the latter few lines, there is a (67,120). If you are magnifying other resolutions, such as 720p, it can be replaced with (45,80).
2. utilities/dataset/py: In the 'Test' class at the end, the __init__ function has a path that needs to be modified. In the subsequent 'transform_in', if your initial input is not 360p, it also needs to be modified.
3. dnn/efficient_det/interface.py: You can search for all places where '1080' or '1920' appear. Currently, it corresponds to 1080p. If you are magnifying to other resolutions, don't hesitate to change it.
4. compress_blackgen_roi.py: Search for 'mask_shape' to find the size that needs to be modified. According to the mask size you want to output, modify it to the corresponding size. As for 'ceil' or 'floor', it's up to you. Note that it corresponds to the size in 'accmpegmodel.py' in item 1.
```

You may encounter an error of 'No such file or directory' when saving the results in subsequent steps. You can try to create two directories `pickles/` and `maskgen_pths/` in advance. In addition, when training, we need to use the Efficientdet model and need to download the weights from the Internet. If the network is not good, the download will be very slow. You can download [Efficientdet weights](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch) in advance from this link and then save it to the path `dnn/efficient_det/weights/`. We use `efficientdet-d0.pth`. If you want to use something else, remember to modify the corresponding part of the code in `dnn/efficient_det/interface.py`.

The training process consists of two stages. First is the generation of ground truth. In this stage, based on the fact that each macroblock has different sensitivities to the accuracy change brought by super-resolution, each macroblock will be assigned different importances. We will obtain a ground truth as a label for subsequent training. The goal of training AccModel is that after inputting a low-resolution image, the region importance distribution output by AccModel is as close as possible to the ground truth. The ground truth will be saved in the `pickles` directory.

After calculating the sensitivity and generating the ground truth, the program will be aborted due to a 'keyError'. However, don't worry. This is a normal phenomenon. Please re-run `batch_train_AccModel.py` to continue training. After the training is completed, the model will be saved in the `maskgen_pths` directory. If you don't want to train AccModel, you can directly skip to step 4 because we have already obtained the ground truth, and only this ground truth is needed for the generalization of the training process.

After completing the training, we hope to use AccModel to predict other video frames and generate the corresponding region importance distribution. Here, we temporarily use the above `input.mp4` for quick understanding of the entire process. First, still modify the data path in `compress_blackgen_roi.py`. Then, to avoid problems caused by inconsistent sizes, you can confirm again whether the size you changed earlier is correct. Finally, running `batch_blackgen_roi.py` can use the trained AccModel to predict and generate the region importance mask. The result will be saved in `importance.pth`.

```python
python batch_train_AccModel.py  ##Train the predictor
python batch_blackgen_roi.py    ##Generate the region importance mask
```

As the code has only been reorganized and maintained recently, if there are problems in the above steps, you can ask questions in **Issues** and @censhallwe. I will answer and update in time! Thanks ♪(･ω･)ﾉ.

## 4. Generalization of the training process

We have tested 6 models of different magnitudes in the article. The above steps 1-3 are the training of the lightweight model AccModel of  [AccMPEG](https://github.com/KuntaiDu/AccMPEG/). In order to further compress the model inference time and improve the model inference efficiency, we have extended the above process to [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg) and tried models of various magnitudes. 

`PaddleSeg` is an end-to-end image segmentation kit completed based on `PaddlePaddle`. It contains a complete image segmentation process including model training, evaluation, prediction, export, and deployment, and contains a large number of available models, which is suitable for beginners to quickly implement segmentation tasks on their own datasets. 

The starting point of this generalization lies in the similarity of tasks. The purpose of the macroblock-based region importance predictor we train is to find the areas in the video where the detection accuracy is more sensitive to super-resolution. Based on the degree of sensitivity, different importances are set for different regions. This is somewhat similar to segmenting an image. We can divide it into different categories according to different degrees of importance. Therefore, we can generalize the above training process to segmentation tasks through `PaddleSeg`. 

For the specific environment configuration and processes such as training, prediction, and exporting ONNX models, you can directly refer to the official documentation of [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg), and we will not repeat them. You can choose a model you are interested in to complete the above process. We mainly introduce the process of dataset preparation. 

In step 4, you will get the region importance mask `ground truth` corresponding to `input.mp4`, which corresponds to the region importance distribution of the corresponding video frame, and the value range is [0, 1]. We obtain the annotation for image segmentation through `ground truth`. First, normalize the `ground truth` to make its value range be [0, 1]. Then, we divide the interval [0, 1] into 10 (or 5, 15, 20, etc.) equal parts and map them to a total of 10 segmentation categories from 0 to 9 respectively (that is, [0, 0.1) corresponds to category 0, [0.1, 0.2) corresponds to category 1, and so on). 

Then, export it by converting it into an image through some methods such as OpenCV. In this way, we can obtain the corresponding annotated image of the input image. Since the original image and the annotated image of `PaddeSeg` need to have the same resolution, the macroblock-level region importance mask `ground truth` needs to be upsampled to the same resolution as the original image. The "nearest" method can be selected for upsampling. 

Then, organize the data as follows. Put the original image in one directory and the annotated image in another directory. Note that the file names should correspond and the suffixes can be different.

```tex
YOUR_dataset
    |
    |--origin_images           # Store all original images
    |  |--image1.png
    |  |--image2.png
    |  |--...
    |
    |--labels                  # Store all label images
    |  |--label1.png
    |  |--label2.png
    |  |--...
```

Finally, split the data. Divide the dataset into training set, validation set, and test set in proportion. Save the relative paths to a TXT file. The information of each line is as follows:

```tex
origin_images/image1.jpg  labels/image1.png
origin_images/image2.jpg  labels/image2.png
...
```

For more details, you can refer to [Preparation of custom datasets for PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.9.1/docs/data/marker/marker_en.md).

## 5. Deploy model

After training the above model, the model can be exported as an ONNX model and then deployed on the server. 

We deploy the model in Nvidia GPU to obtain faster computing speed. The adopted method is TensorRT computing method. You need to prepare the corresponding environment. You can download and install the TensorRT version compatible with your CUDA and cudnn on [TensorRT official website](https://developer.nvidia.com/tensorrt). 

After that, model deployment can be completed through `onnx2trt.py` and `trt_infer.py`.

## Acknowledgements

Thanks for the help of the following work: [AccMPEG](https://github.com/KuntaiDu/AccMPEG/)，[EDSR](https://github.com/sanghyun-son/EDSR-PyTorch)，[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg)，[TensorRT](https://github.com/NVIDIA/TensorRT)















