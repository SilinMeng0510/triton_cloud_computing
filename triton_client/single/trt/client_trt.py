# Figure 3: different model types, QPS

from tritonclient.utils import triton_to_np_dtype
from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput
from transformers import AutoTokenizer
import numpy as np
import time
import socket
import seaborn as sns
import matplotlib.pyplot as plt
import json

def send_signal(host='127.0.0.1', port=12345):
    connected = False
    while not connected:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((host, port))
                s.sendall(b'Client is Ready.')
                connected = True  # 连接成功，跳出循环
        except ConnectionRefusedError:
            print("Connection refused. Server might be down. Retrying in 5 seconds...")
            time.sleep(5)  # 等待5秒再次尝试连接

NUM_SAMPLE = 10000
model_name = "albert_mbti_trt"

tokenizer = AutoTokenizer.from_pretrained('JanSt/albert-base-v2_mbti-classification')

url = "localhost:8000"
client = InferenceServerClient(url=url)

# 数据预准备 （重复利用同一数据）
text = "In order to have great happiness you have to have great pain and unhappiness - otherwise how would you know when you're happy?"
inputs = tokenizer(text, return_tensors="np", padding='max_length', truncation=True, max_length=512)

input_ids = inputs["input_ids"].astype(np.int32)
attention_mask = inputs["attention_mask"].astype(np.int32)
token_type_ids = inputs.get("token_type_ids", np.zeros_like(input_ids)).astype(np.int32)

input0 = InferInput('input_ids', input_ids.shape, 'INT32')
input1 = InferInput('attention_mask', attention_mask.shape, 'INT32')
input2 = InferInput('token_type_ids', token_type_ids.shape, 'INT32')
output = InferRequestedOutput('logits')

input0.set_data_from_numpy(input_ids)
input1.set_data_from_numpy(attention_mask)
input2.set_data_from_numpy(token_type_ids)

# 设备预热
print("Warming up...")
for i in range(100):
    # 数据预处理
    inputs = tokenizer(text, return_tensors="np", padding='max_length', truncation=True, max_length=512)


    # 发送推理请求
    response = client.infer(model_name=model_name,
                            inputs=[input0, input1, input2],
                            outputs=[output])

# 发送信号给utilization.py，收集GPU和CPU数据
send_signal()

# 基准测试
record_preprocess_latency = []
record_inference_latency = []

for i in range(NUM_SAMPLE):
    # 数据预处理
    start_preprocess_time = time.time()
    inputs = tokenizer(text, return_tensors="np", padding='max_length', truncation=True, max_length=512)
    preprocess_latency = time.time() - start_preprocess_time
    record_preprocess_latency.append(preprocess_latency)
    print(f"SAMPLE-{i} Preprocessing latency: {preprocess_latency} seconds")

    # 发送推理请求
    start_inference_time = time.time()
    response = client.infer(model_name=model_name,
                            inputs=[input0, input1, input2],
                            outputs=[output])
    inference_latency = time.time() - start_inference_time
    record_inference_latency.append(inference_latency)
    print(f"SAMPLE-{i} Inference latency: {inference_latency} seconds")
    
    # logits = response.as_numpy('logits')
    # print(f"SAMPLE-{i} Inference results:", logits)
    print("-----------------------------------------------------------")

array_preprocess_latency = np.array(record_preprocess_latency)
array_inference_latency = np.array(record_inference_latency)

with open('plot/single/preprocess_latency_single.json', 'w') as file:
    json.dump(record_preprocess_latency, file)
    
with open('plot/single/inference_latency_single.json', 'w') as file:
    json.dump(record_inference_latency, file)

p50_preprocess, p90_preprocess, p99_preprocess = np.percentile(array_preprocess_latency, 50), np.percentile(array_preprocess_latency, 90), np.percentile(array_preprocess_latency, 99)
print(f"Total Preprocessing latency: {np.sum(array_preprocess_latency)} seconds")
print(f"P50 Preprocessing latency: {p50_preprocess} seconds")
print(f"P90 Preprocessing latency: {p90_preprocess} seconds")
print(f"P99 Preprocessing latency: {p99_preprocess} seconds")
print(f"Average Preprocessing latency: {np.sum(array_preprocess_latency) / NUM_SAMPLE} seconds")

with open('plot/single/preprocess_P_single.json', 'w') as file:
    json.dump({'p50' : p50_preprocess, 'p90' : p90_preprocess, 'p99' : p99_preprocess}, file)

# 绘制预处理KDE图并保存
sns.kdeplot(record_preprocess_latency)
plt.title('KDE of Preprocessing Latencies')
plt.xlabel('Latency')
plt.ylabel('Density')
plt.savefig('single/trt/outcome/PREPROCESS_KDE.png')
plt.close() 

# 绘制预处理CDF图
preprocess_latency_sorted = np.sort(record_preprocess_latency)
cdf = np.arange(1, len(preprocess_latency_sorted) + 1) / len(preprocess_latency_sorted)
plt.plot(preprocess_latency_sorted, cdf)
plt.title('CDF of Preprocessing Latencies')
plt.xlabel('Latency')
plt.ylabel('CDF')
plt.grid(True)
plt.savefig('single/trt/outcome/PREPROCESS_CDF.png')
plt.close()

print("**********************************************************")

p50_inference, p90_inference, p99_inference = np.percentile(array_inference_latency, 50), np.percentile(array_inference_latency, 90), np.percentile(array_inference_latency, 99)
print(f"Total Inference latency: {np.sum(array_inference_latency)} seconds")
print(f"P50 Inference latency: {p50_inference} seconds")
print(f"P90 Inference latency: {p90_inference} seconds")
print(f"P99 Inference latency: {p99_inference} seconds")
print(f"Average Inference latency: {np.sum(array_inference_latency) / NUM_SAMPLE} seconds")

with open('plot/single/inference_P_single.json', 'w') as file:
    json.dump({'p50' : p50_inference, 'p90' : p90_inference, 'p99' : p99_inference}, file)

# 绘制推理KDE图并保存
sns.kdeplot(record_inference_latency)
plt.title('KDE of Inference Latencies')
plt.xlabel('Latency (s)')
plt.ylabel('Density')
plt.savefig('single/trt/outcome/INFERENCE_KDE.png')
plt.close() 

# 绘制推理CDF图
inference_latency_sorted = np.sort(record_inference_latency)
cdf = np.arange(1, len(inference_latency_sorted) + 1) / len(inference_latency_sorted)
plt.plot(inference_latency_sorted, cdf)
plt.title('CDF of Inference Latencies')
plt.xlabel('Latency (s)')
plt.ylabel('CDF (latency) [%]')
plt.grid(True)
plt.savefig('single/trt/outcome/INFERENCE_CDF.png')
plt.close()

print("**********************************************************")

# 这里的QPS包含了预处理时间和模型推理时间 (无后处理时间)
print(f"Throughput(QPS): {NUM_SAMPLE / (np.sum(array_preprocess_latency) + np.sum(array_inference_latency))} seconds")

