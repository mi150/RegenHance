import cv2
import onnxruntime as ort
import numpy as np
import torchvision.io as io

onnx_file_model = 'output_cova_mb2_new/mb2_compressive.onnx'
onnx_file_model = 'HarDNet_seg_bs_32_360p.onnx'
onnx_file_model = 'Acc_seg_model_onnx/Acc_seg_dynamic_bs_360p.onnx'
onnx_file_model = 'hardnet_onnx_model/HarDNet_seg_dy_bs_dy_hw.onnx'
onnx_file_model = './EDSR_one_block.onnx'
onnx_file_model = './prune_EDSR/EDSR_x2.onnx'
onnx_file_model = 'dy_EDSR_test.onnx'
onnx_file_model = 'hardnet_onnx_model/HarDNet_seg_dy_bs_360p.onnx'
onnx_file_model = './InputSizeEDSR/EDSR_16.onnx'
# 找到 GPU / CPU
provider = ort.get_available_providers()[1 if ort.get_device() == 'GPU' else 0]
#provider1 = ort.get_available_providers()[2]
print('设备:', provider)
# 声明 onnx 模型
model = ort.InferenceSession(onnx_file_model , providers=[provider])
 
# 参考: ort.NodeArg
for node_list in model.get_inputs(), model.get_outputs():
    for node in node_list:
        attr = {'name': node.name,
                'shape': node.shape,
                'type': node.type}
        print(attr)
    print('-' * 80)

# 得到输入、输出结点的名称
input_node_name = model.get_inputs()[0].name
ouput_node_name = [node.name for node in model.get_outputs()]
'''
image = io.read_image('/home/ubuntu/csw/PaddleSeg/data/video_test/decode/frame-0000.jpg').unsqueeze(dim=0)
image = (image/255.0).numpy()
print(image.shape)
print(image)
output = model.run(output_names=ouput_node_name, input_feed={input_node_name: image})
print(output[0].shape)
#output = output[0][0,:,:,:]
#output_ = np.argmax(output,axis=0)
res = output[0]
print(res)
result_mask = (res.transpose(1, 2, 0)*255).astype(np.uint8)
cv2.imwrite("test_result.jpg", result_mask)
'''
