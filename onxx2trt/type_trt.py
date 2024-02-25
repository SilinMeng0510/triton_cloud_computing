import tensorrt as trt

# 指定你的 .plan 文件路径
TRT_ENGINE_PATH = 'albert-base-v2_mbti-classification.plan'

# 创建一个TensorRT logger对象
TRT_LOGGER = trt.Logger(trt.Logger.INFO)

# 加载TensorRT引擎
def load_engine(trt_engine_path):
    with open(trt_engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

# 打印输入和输出绑定的信息
def print_bindings_info(engine):
    for binding_index in range(engine.num_bindings):
        binding_name = engine.get_binding_name(binding_index)
        binding_shape = engine.get_binding_shape(binding_index)
        binding_dtype = engine.get_binding_dtype(binding_index)
        binding_type = "Input" if engine.binding_is_input(binding_index) else "Output"
        print(f"{binding_type} Binding '{binding_name}': Shape={binding_shape}, Dtype={binding_dtype}")

# 主程序
if __name__ == '__main__':
    engine = load_engine(TRT_ENGINE_PATH)
    if engine:
        print_bindings_info(engine)
    else:
        print("Failed to load engine.")
