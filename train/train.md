[🇺🇸](/train/train.md "English") [🇨🇳](/train/train-cn.md "Simplified Chinese")

# Train a macroblock-based region importance predictor

This README file will guide you step by step to implement training a macroblock-level region importance predictor on your own dataset. The training code has not been maintained for a long time and some bugs may be encountered. If you encounter any problems at following steps, you are welcome to leave a message and ask questions in **[Issues](https://github.com/mi150/RegenHance)**. ❤️

## 1. Set up the environment

First, git clone our repo to your working directory $DIR. Then, install the corresponding conda environment through the `conda_env.yml` file:

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

During the training process, in order to calculate region importance, we need blurred input and clear input. 

The blurred input is obtained by interpolating and enlarging the above png images (for example, interpolating to 1080p to enlarge by 3 times). This process can be completed in advance and saved to a specific directory or embedded in the training process, because the interpolation calculation is relatively simple and fast. 

The clear input is completed through super-resolution. You need to prepare a trained super-resolution model in advance. We use the [EDSR](https://github.com/sanghyun-son/EDSR-PyTorch) super-resolution model. Other models are also OK. Just configure different models according to the multiple of resolution you wish to improve. Since full-frame super-resolution has a large amount of computation, we recommend completing the super-resolution of low-resolution images in advance before training the macroblock-based region importance predictor and saving them in a specific directory. 

After completing this step, organize your dataset into a structure similar to the following:

```tex
YOUR_dataset
    |
    |--origin_images       # Store all original images or interpolated blurred images.
    |  |--0000000000.png
    |  |--0000000001.png
    |  |--...
    |
    |--sr_images           # Store all super-resolution clear images.
    |  |--0000000000.png
    |  |--0000000001.png
    |  |--...
```

## 3. Train model



## 4. Generalization of the training process

We have tested 6 models of different magnitudes in the article. The above steps 1-3 are the training of the lightweight model AccModel of  [AccMPEG](https://github.com/KuntaiDu/AccMPEG/). In order to further compress the model inference time and improve the model inference efficiency, we have extended the above process to [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg) and tried models of various magnitudes. 

`PaddleSeg` is an end-to-end image segmentation kit completed based on `PaddlePaddle`. It contains a complete image segmentation process including model training, evaluation, prediction, export, and deployment, and contains a large number of available models, which is suitable for beginners to quickly implement segmentation tasks on their own datasets. 

The starting point of this generalization lies in the similarity of tasks. The purpose of the macroblock-based region importance predictor we train is to find the areas in the video where the detection accuracy is more sensitive to super-resolution. Based on the degree of sensitivity, different importances are set for different regions. This is somewhat similar to segmenting an image. We can divide it into different categories according to different degrees of importance. Therefore, we can generalize the above training process to segmentation tasks through `PaddleSeg`. 

For the specific environment configuration and processes such as training, prediction, and exporting ONNX models, you can directly refer to the official documentation of [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg), and we will not repeat them. You can choose a model you are interested in to complete the above process. We mainly introduce the process of dataset preparation. 

In step 4, you will get the region importance mask `importance` corresponding to `input.mp4`, which corresponds to the region importance distribution of the corresponding video frame, and the value range is [0, 1]. We obtain the annotation for image segmentation through `importance`. We divide the interval [0,1] into 10 (or 5, 15, 20, etc.) equal parts and map them to a total of 10 segmentation categories from 0 to 9 respectively (that is, [0, 0.1) corresponds to category 0, [0.1, 0.2) corresponds to category 1, and so on). 

Then, export it by converting it into an image through some methods such as OpenCV. In this way, we can obtain the corresponding annotated image of the input image. Since the original image and the annotated image of `PaddeSeg` need to have the same resolution, the macroblock-level region importance mask `importance` needs to be upsampled to the same resolution as the original image. The "nearest" method can be selected for upsampling. 

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















