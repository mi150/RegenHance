[🇺🇸](/train.md "English") [🇨🇳](/train-cn.md "Simplified Chinese")

# 宏块级区域重要性预测器训练

这个README文件将带你一步步实现训练一个自己数据集上的宏块级区域重要性预测器，训练所用代码太久没有维护，可能会遇到一些bug，如果在过程中遇到任何问题，欢迎在 **[Issues](https://github.com/mi150/RegenHance/issues)** 中留言提问❤

## 1. 准备环境

首先，git clone我们的repo到你的工作目录$DIR下，然后通过`conda_env.yml`文件安装相应的conda环境，如果pip安装某些库时出错，可以尝试换源（比如清华源）后通过`train_AccModel/requirements.txt`再去下载：

```bash
git clone https://github.com/mi150/RegenHance.git
conda env create -f conda_env.yml
```

然后激活安装好的环境：

```bash
conda activate regenhance
```

在配置好环境后，请通过 **`pip`** 方式安装`pytorch`和`torchvision`（可以参考该链接：[Pytorch](https://pytorch.org/get-started/locally/)，我们使用的pytorch=1.10.1，CUDA环境为11.1），还需安装`detectron2`（可参考[detectron2](https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md)，注意匹配正确的pytorch和CUDA版本）。

上述过程会自动安装一个旧版本的`ffmpeg`，需要用新版本进行替换，请从[FFmpeg](https://johnvansickle.com/ffmpeg/)中下载一个static版本的ffmpeg并用其对原本的ffmpeg进行替换（你可以通过`which ffmpeg`命令找到原先的ffmpeg，我们所使用的ffmpeg版本为5.0.1）。

由于后续准备数据时需要进行超分，你可以选用合适的超分模型，我们选用了`EDSR`模型作为超分模型，具体环境配置可参考[EDSR](https://github.com/sanghyun-son/EDSR-PyTorch)官方github。

## 2. 准备数据

回到你的工作目录$DIR下，准备好你自己希望训练的视频数据，我们提供了一个`input.mp4`作为参考，以车辆和行人检测为例，复现训练过程。

运行`extract.py`从`input.mp4`视频中提取出png格式的图片，这些图片将作为360p低分辨率下的视频原始输入。

在训练过程中，为了计算区域重要性，我们需要模糊的输入和清晰的输入。模糊的输入就是上述的低分辨率图像。清晰的输入通过超分完成，你需要提前准备一个训练好的超分模型，我们使用的是[EDSR](https://github.com/sanghyun-son/EDSR-PyTorch)超分模型，其他模型也可以，根据你希望提升的分辨率倍数配置不同的模型即可。由于整帧超分计算量较大，我们建议在训练宏块级区域重要性预测器过程前提前完成对低分辨图片的超分，保存在特定的目录下。

完成该步骤后，将你的数据集整理成类似如下的结构：

```tex
YOUR_dataset
    |
    |--or_images           # 存放所有低分辨率的原图
    |  |--0000000000.png
    |  |--0000000001.png
    |  |--...
    |
    |--sr_images           # 存放所有超分后的清晰图像
    |  |--0000000000.png
    |  |--0000000001.png
    |  |--...
```

## 3. 模型训练

做好上述准备后，进入$DIR/train_AccModel目录下，修改`train_AccModel.py`文件中的数据集路径，之后运行`batch_train_AccModel.py`开始训练宏块级区域重要性预测器（目前代码默认是360p超分至1080p，即3倍的情况下进行训练，如果是其他尺寸和放大倍数，请修改下面几个文件中的尺寸大小）。

```tex
1. maskgen/SSD/accmpegmodel.py：后面几行中有个(67,120)，如果是放大其他分辨率，比如720p，就可以替换为(45,80)。
2. utilities/dataset/py：最后Test类中，__init__函数有个路径需要修改，后面的transform_in中，如果你的初始输入不是360p的话，也需要进行修改。
3. dnn/efficient_det/interface.py：你可以搜索所有出现1080或1920的地方，目前对应的是1080p，如果你是放大到其他分辨率，别犹豫，改它。
4. compress_blackgen_roi.py：搜索mask_shape可找到需要修改的尺寸，针对你想输出的mask尺寸，修改成对应的大小即可，注意和1中的accmpegmodel.py里的尺寸对应，至于上取整还是下取整都可以，看你的想法。
```

可能在后续步骤保存结果时你会遇到找不到路径的错误，可以尝试提前创建两个目录`pickles/`和`maskgen_pths/`，此外在训练时我们需要用到Efficientdet模型，需要从网上下载权重，如果网络不好的话，下载会很慢，可以提前从这个链接中下载[Efficientdet权重](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch)，然后保存到`dnn/efficient_det/weights/`这个路径下，我们用的是`efficientdet-d0.pth`，如果你想用别的，记得修改`dnn/efficient_det/interface.py`对应部分的代码。

训练过程包含两个阶段，首先是生成ground truth，在这一阶段中，基于各宏块对超分带来的精度变化的敏感程度不同，各个宏块将被分配到不同的重要性，我们会得到一个ground truth作为后续训练的标签，我们训练AccModel的目标就是输入低分辨率图像后，模型输出的区域重要性分布于ground truth尽可能接近。ground truth会保存在`pickles`目录下。

在计算完敏感程度生成ground truth后程序会由于keyError而中止，不过不要担心，这是正常现象，请重新运行`batch_train_AccModel.py`继续训练，训练完成后，模型将保存在`maskgen_pths`目录中。如果你不想训练AccModel，可直接跳至第4步，因为我们已经得到了ground truth，而训练过程的推广只需要用到这个ground truth即可。

在完成训练后，我们希望使用AccModel去对其他视频帧进行预测，生成对应的区域重要性分布。这里我们暂时先沿用上面的`input.mp4`以便快速理解整个流程。首先还是先修改`compress_blackgen_roi.py`中的数据路径。然后为了避免因尺寸不一致导致的问题，可以再确认一下前面改的尺寸是否正确。最后运行`batch_blackgen_roi.py`可使用训练好的AccModel预测生成区域重要性掩码，结果将会保存至`importance.pth`中。

```python
python batch_train_AccModel.py  ##训练预测器
python batch_blackgen_roi.py    ##生成区域重要性掩码
```

由于最近才重新整理维护代码，如果上述步骤出现问题，可以在 **Issues** 提问并@censhallwe，我会及时解答并更新！感谢♪(･ω･)ﾉ

## 4. 训练过程的推广

我们在文中使用了6种不同量级的模型进行了测试，上述1-3步是轻量级模型[AccMPEG](https://github.com/KuntaiDu/AccMPEG/)的AccModel的训练，为了进一步压缩模型推理时间，提升模型推理效率，我们将上述过程推广到了[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg)，尝试了多种量级的模型。`PaddleSeg`是基于`飞桨PaddlePaddle`完成的端到端图像分割套件，包含模型训练、评估、预测、导出、部署等完整图像分割流程，且包含大量可用模型，适合新手快速实现在自己数据集上的分割任务。

这个推广的出发点在于任务的相似性。我们训练的宏块级区域重要性预测器目的是找出视频中检测精度受超分影响较为敏感的区域，基于敏感程度对不同区域设定了不同的重要性，这在某种程度上类似于对图像进行分割，根据重要性程度的不同分割成不同的类别。因此，我们可以通过`PaddleSeg`将上述训练过程推广到分割任务中。

具体环境配置以及训练、预测、导出ONNX模型等过程可直接参考[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg)官方文档，我们便不再赘述，你可以选用自己感兴趣的模型完成上述流程。我们主要介绍数据集准备的过程。

在第4步中，你会得到`input.mp4`对应的区域重要性掩码`ground truth`，其对应相应视频帧的区域重要性分布。我们通过`ground truth`来获得图像分割的标注，首先对`ground truth`进行归一化，使其值域为[0, 1]，然后我们将[0,1]的区间10等分（或者5,15,20等），分别映射到0-9共十个分割类别中（即[0, 0.1)对应类别0，[0.1, 0.2)对应类别1，依次类推），之后通过OpenCV等方式将其转换成图片导出，这样我们就能获得输入图像相应的标注图像。由于`PaddleSeg`的原图和标注图像需要分辨率相同，所以需要将宏块级的区域重要性掩码`ground truth`上采样至和原图相同的分辨率，上采样选择“最邻近”方式即可。

然后，将数据整理如下结构，将原图放在一个目录下，标注图像放在另一个目录下，注意文件名要对应，后缀名可以不同。

```tex
YOUR_dataset
    |
    |--origin_images    # 存放所有原图
    |  |--image1.png
    |  |--image2.png
    |  |--...
    |
    |--labels           # 存放所有标注图
    |  |--label1.png
    |  |--label2.png
    |  |--...
```

最后，切分数据，将数据集按比例划分为训练集、验证集、测试集，将相对路径保存至TXT文件中，每一行信息如下：

```tex
origin_images/image1.jpg  labels/image1.png
origin_images/image2.jpg  labels/image2.png
...
```

更多细节可以参考[PaddleSeg自定义数据集准备](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.9.1/docs/data/marker/marker_cn.md)。

## 5.模型部署

在训练完上述模型后，可以将模型导出成ONNX模型，再将其部署在服务器上。

我们将模型部署至Nvidia GPU中以获得更快的计算速度，采用的方式是TensorRT计算方式，需要准备相应的环境，大家可以在[TensorRT官网](https://developer.nvidia.com/tensorrt)下载安装和自己CUDA与cudnn适配的TensorRT版本。

之后可通过的`onnx2trt.py`和`trt_infer.py`完成模型部署。

## 致谢

感谢以下工作的帮助：[AccMPEG](https://github.com/KuntaiDu/AccMPEG/)，[EDSR](https://github.com/sanghyun-son/EDSR-PyTorch)，[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg)，[TensorRT](https://github.com/NVIDIA/TensorRT)
