import torch
import tensorrt as trt
from torch2trt import TRTModule
from transformers import AutoTokenizer
engine_file_path = "./albert-base-v2_mbti-classification.plan"
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
logger = trt.Logger(trt.Logger.INFO)
model_all_names = []
with open(engine_file_path, "rb") as f, trt.Runtime(logger) as runtime:
    engine=runtime.deserialize_cuda_engine(f.read())
for binding in engine:
        size = trt.volume(engine.get_tensor_shape(binding)) * 1
        dims = engine.get_tensor_shape(binding)
        # print(size)
        print(dims)
        print(binding)
        print("input =", engine.get_tensor_shape(binding))
        dtype = trt.nptype(engine.get_tensor_dtype(binding))
     
trt_model = TRTModule(engine, ["input_ids", "attention_mask", "token_type_ids"],['logits']).to(device)
sentence = "Hello， how are you"
tokenizer = AutoTokenizer.from_pretrained("JanSt/albert-base-v2_mbti-classification")

data = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=512)
input_ids = data['input_ids'].int().to(device)
attention_mask = data['attention_mask'].int().to(device)
token_type_ids = data['token_type_ids'].int().to(device)
out = trt_model(input_ids,attention_mask,token_type_ids)
print(out)

   