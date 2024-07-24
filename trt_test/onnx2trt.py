import os
import tensorrt as trt
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#ONNX_SIM_MODEL_PATH = 'output_cova_mb2_new/mb2_compressive.onnx'
#TENSORRT_ENGINE_PATH_PY = 'output_cova_mb2_new/mb2_compressive.engine'
# ONNX_SIM_MODEL_PATH = '/home/dodo/Downloads/vit_l_16_bs4.onnx'
# TENSORRT_ENGINE_PATH_PY = 'vit_l_16_bs4.engine'
ONNX_SIM_MODEL_PATH='HarDNet_seg_dy_bs_dy_input_hw.onnx'
TENSORRT_ENGINE_PATH_PY = 'hardnet.engine'
ONNX_SIM_MODEL_PATH='./InputSizeEDSR/EDSR_1024.onnx'
TENSORRT_ENGINE_PATH_PY = './InputSizeEDSR/EDSR_1024.engine'
ONNX_SIM_MODEL_PATH='mask-rcnn-2ef0cd.onnx'
TENSORRT_ENGINE_PATH_PY = 'swin.engine'
ONNX_SIM_MODEL_PATH='./DETR/detr-s.onnx'
TENSORRT_ENGINE_PATH_PY = './DETR/detr-s.engine'
# nms_plugin = trt.get_plugin_registry().create_plugin("NMSPluginDynamic", plugin_version="1")
# max_batches = 1  # 最大批次数
# score_threshold = 0.5  # 类别得分阈值
# iou_threshold = 0.3  # IoU阈值
# top_k = 100  # 保留的边界框数量
# background_class = 0  # 背景类别索引
# nms_plugin.set_plugin_namespace("")  # 设置插件命名空间
# nms_plugin.set_input_score_threshold(score_threshold)
# nms_plugin.set_input_iou_threshold(iou_threshold)
# nms_plugin.set_input_top_k(top_k)
# nms_plugin.set_input_background_class(background_class)

def build_engine(onnx_file_path, engine_file_path, flop=16):
    trt_logger = trt.Logger(trt.Logger.VERBOSE)  # trt.Logger.ERROR
    builder = trt.Builder(trt_logger)
    network = builder.create_network(
        1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    # output_layer=network.get_layer(network.num_layers-1)
    # nms_layer = network.add_plugin_v2(inputs=[output_layer.get_output(0)], plugin=nms_plugin)
    # nms_layer.get_output(0).name = "nms_output"
    parser = trt.OnnxParser(network, trt_logger)
    # parse ONNX
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    print("Completed parsing ONNX file")
    #builder.max_workspace_size = 2 << 30
    config = builder.create_builder_config()
    config.max_workspace_size = 2 << 32
    # default = 1 for fixed batch size
    builder.max_batch_size = 1
    # set mixed flop computation for the best performance
    
    if builder.platform_has_fast_fp16 and flop == 16:
        #builder.fp16_mode = True
        config.flags |= 1<<int(trt.BuilderFlag.TF32)

    if os.path.isfile(engine_file_path):
        try:
            os.remove(engine_file_path)
        except Exception:
            print("Cannot remove existing file: ",
                engine_file_path)

    print("Creating Tensorrt Engine")

    #config = builder.create_builder_config()
    config.set_tactic_sources(1 << int(trt.TacticSource.CUBLAS))
    config.max_workspace_size = 2 << 32
    config.set_flag(trt.BuilderFlag.TF32)
    
    engine = builder.build_engine(network, config)
    with open(engine_file_path, "wb") as f:
        f.write(engine.serialize())
    print("Serialized Engine Saved at: ", engine_file_path)
    return engine




def build_engine_1(
                 onnx_file_path,
                 engine_file_path,
                 *,
                 use_fp16=True,
                 dynamic_shapes={},
                 dynamic_batch_size=1):
    """Build TensorRT Engine
    :use_fp16: set mixed flop computation if the platform has fp16.
    :dynamic_shapes: {binding_name: (min, opt, max)}, default {} represents not using dynamic.
    :dynamic_batch_size: set it to 1 if use fixed batch size, else using max batch size
    """
    trt_logger = trt.Logger(trt.Logger.VERBOSE)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(
        1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

    config = builder.create_builder_config()
    config.set_tactic_sources(trt.TacticSource.CUBLAS_LT)

    # Default workspace is 2G
    config.max_workspace_size = 2 << 32

    if builder.platform_has_fast_fp16 and use_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # parse ONNX
    parser = trt.OnnxParser(network, trt_logger)
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    print("===> Completed parsing ONNX file")

    # default = 1 for fixed batch size
    builder.max_batch_size = 1

    if len(dynamic_shapes) > 0:
        print(f"===> using dynamic shapes: {str(dynamic_shapes)}")
        builder.max_batch_size = dynamic_batch_size
        profile = builder.create_optimization_profile()

        for binding_name, dynamic_shape in dynamic_shapes.items():
            min_shape, opt_shape, max_shape = dynamic_shape
            profile.set_shape(
                binding_name, min_shape, opt_shape, max_shape)

        config.add_optimization_profile(profile)

    # Remove existing engine file
    if os.path.isfile(engine_file_path):
        try:
            os.remove(engine_file_path)
        except Exception:
            print(f"Cannot remove existing file: {engine_file_path}")

    print("===> Creating Tensorrt Engine...")
    engine = builder.build_engine(network, config)
    if engine:
        with open(engine_file_path, "wb") as f:
            f.write(engine.serialize())
        print("===> Serialized Engine Saved at: ", engine_file_path)
    else:
        print("===> build engine error")
    return engine


if __name__ == "__main__":    
    dynamic_shapes = {"input": ((1, 3, 800, 1333), (1, 3, 800, 1333),(1, 3, 800, 1333))}
    build_engine_1(ONNX_SIM_MODEL_PATH, TENSORRT_ENGINE_PATH_PY,dynamic_shapes=dynamic_shapes)
    #build_engine(ONNX_SIM_MODEL_PATH, TENSORRT_ENGINE_PATH_PY)

