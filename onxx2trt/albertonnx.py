from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import torch

# 加载模型和分词器
model_name = "JanSt/albert-base-v2_mbti-classification"
model = AutoModelForSequenceClassification.from_pretrained(model_name)



tokenizer = AutoTokenizer.from_pretrained(model_name)

text = "This is a sample text for ONNX conversion."
inputs = tokenizer(text, return_tensors="pt")

inputs_int32 = {key: value.to(torch.int32) for key, value in inputs.items()}

print(inputs.values())
print(inputs_int32.values())

# 使用torch.onnx.export进行转换
torch.onnx.export(model, 
                  args=tuple(inputs_int32.values()), 
                  f="albert-base-v2_mbti-classification.onnx", 
                  input_names=['input_ids', 'token_type_ids', 'attention_mask'], 
                  output_names=["logits"], 
                  dynamic_axes={'input_ids': {0: 'batch_size', 1: 'sequence'}, 
                                'token_type_ids': {0: 'batch_size', 1: 'sequence'}, 
                                'attention_mask': {0: 'batch_size', 1: 'sequence'}, 
                                'logits': {0: 'batch_size', 1: 'sequence'}}, 
                  do_constant_folding=True, 
                  opset_version=11) 
