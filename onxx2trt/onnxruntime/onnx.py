import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer

# 加载ONNX模型
onnx_model_path = "albert-base-v2_mbti-classification.onnx"
ort_session = ort.InferenceSession(onnx_model_path)

# 加载相应的分词器
tokenizer = AutoTokenizer.from_pretrained("JanSt/albert-base-v2_mbti-classification")

# 准备输入数据
text = "Hello， how are you"
inputs = tokenizer(text, return_tensors="np")  # 使用return_tensors="np"来获取NumPy张量
inputs = {key: value.astype(np.int32) for key, value in inputs.items()}
input_dict = {k: v for k, v in inputs.items()}

print(input_dict)
print(ort_session.get_inputs()[0])

# 执行ONNX模型推理
ort_inputs = {ort_session.get_inputs()[i].name: input_dict[name] for i, name in enumerate(input_dict)}
ort_outputs = ort_session.run(None, ort_inputs)

# 输出推理结果
print(ort_outputs)