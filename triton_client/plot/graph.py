import json
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# 多租户数据
with open('plot/multiple/multiple_runtime.json', 'r') as file:
    multiple_runtime = json.load(file)
    print(len(multiple_runtime))

with open('plot/multiple/cpu_util_multiple.json', 'r') as file:
    cpu_util_multiple = json.load(file)
    
with open('plot/multiple/gpu_util_multiple.json', 'r') as file:
    gpu_util_multiple = json.load(file)
    
with open('plot/multiple/inference_latency_multiple.json', 'r') as file:
    inference_latency_multiple = json.load(file)
    inference_latency_multiple = [value * 1000 for value in inference_latency_multiple]
    
with open('plot/multiple/preprocess_latency_multiple.json', 'r') as file:
    preprocess_latency_multiple = json.load(file)
    preprocess_latency_multiple = [value * 1000 for value in preprocess_latency_multiple]

with open('plot/multiple/inference_P_multiple.json', 'r') as file:
    inference_P_multiple = json.load(file)
    inference_P_multiple = {key: value * 1000 for key, value in inference_P_multiple.items()}

with open('plot/multiple/preprocess_P_multiple.json', 'r') as file:
    preprocess_P_multiple = json.load(file)
    preprocess_P_multiple = {key: value * 1000 for key, value in preprocess_P_multiple.items()}

# 单租户数据
with open('plot/single/single_runtime.json', 'r') as file:
    single_runtime = json.load(file)
    print(len(single_runtime))

with open('plot/single/cpu_util_single.json', 'r') as file:
    cpu_util_single = json.load(file)
    
with open('plot/single/gpu_util_single.json', 'r') as file:
    gpu_util_single = json.load(file)
    
with open('plot/single/inference_latency_single.json', 'r') as file:
    inference_latency_single = json.load(file)
    inference_latency_single = [value * 1000 for value in inference_latency_single]
    
with open('plot/single/preprocess_latency_single.json', 'r') as file:
    preprocess_latency_single = json.load(file)
    preprocess_latency_single = [value * 1000 for value in preprocess_latency_single]

with open('plot/single/inference_P_single.json', 'r') as file:
    inference_P_single = json.load(file)
    inference_P_single = {key: value * 1000 for key, value in inference_P_single.items()}

with open('plot/single/preprocess_P_single.json', 'r') as file:
    preprocess_P_single = json.load(file)
    preprocess_P_single = {key: value * 1000 for key, value in preprocess_P_single.items()}

# 数据调整
gpu_util_multiple = gpu_util_multiple[:len(gpu_util_single)]
cpu_util_multiple = cpu_util_multiple[:len(cpu_util_single)]
    
# GPU utilization
plt.figure(figsize=(10, 5))

plt.plot(single_runtime, gpu_util_multiple, label='Single Tenant and Single Instance')
plt.plot(single_runtime, gpu_util_single, label='Multiple Tenants and Multiple Instances')

plt.title('GPU Utilization Over Time')
plt.xlabel('Runtime')
plt.ylabel('Utilization (%)')

plt.legend(loc='lower right')
plt.savefig(f"plot/outcome/GPU_UTIL.pdf")
plt.close()

# CPU utilization
plt.figure(figsize=(10, 5))

plt.plot(single_runtime, cpu_util_multiple, label='Single Tenant and Single Instance')
plt.plot(single_runtime, cpu_util_single, label='Multiple Tenants and Multiple Instances')

plt.title('CPU Utilization Over Time')
plt.xlabel('Runtime')
plt.ylabel('Utilization (%)')

plt.legend(loc='lower right')
plt.savefig(f"plot/outcome/CPU_UTIL.pdf")
plt.close()

# Inference KDE
sns.kdeplot(inference_latency_single, color="skyblue", fill=True, label='Single Tenant and Single Instance')
sns.kdeplot(inference_latency_multiple, color="orange", fill=True, label='Multiple Tenants and Multiple Instances')

plt.title('KDE of Inference Latencies')
plt.xlabel('Latency (ms)')
plt.ylabel('Density')

plt.legend(loc='upper right')
plt.savefig('plot/outcome/INFERENCE_KDE.pdf')
plt.close() 

# Inference CDF
inference_latency_single_sorted = np.sort(inference_latency_single)
single_cdf = np.arange(1, len(inference_latency_single_sorted) + 1) / len(inference_latency_single_sorted)
idxs_p90 = np.searchsorted(inference_latency_single_sorted, inference_P_single['p90'], side='right') - 1
idxs_p99 = np.searchsorted(inference_latency_single_sorted, inference_P_single['p99'], side='right') - 1

inference_latency_multiple_sorted = np.sort(inference_latency_multiple)
multiple_cdf = np.arange(1, len(inference_latency_multiple_sorted) + 1) / len(inference_latency_multiple_sorted)
idxm_p90 = np.searchsorted(inference_latency_multiple_sorted, inference_P_multiple['p90'], side='right') - 1
idxm_p99 = np.searchsorted(inference_latency_multiple_sorted, inference_P_multiple['p99'], side='right') - 1

plt.plot([inference_P_multiple['p90'], inference_P_multiple['p90']], [0, multiple_cdf[idxm_p90]], color='cyan', linestyle='--', linewidth=1, label='P90 Latency')
plt.plot([inference_P_single['p90'], inference_P_single['p90']], [0, single_cdf[idxs_p90]], color='cyan', linestyle='--', linewidth=1)
plt.plot([inference_P_multiple['p99'], inference_P_multiple['p99']], [0, multiple_cdf[idxm_p99]], color='orange', linestyle='--', linewidth=1, label='P99 Latency')
plt.plot([inference_P_single['p99'], inference_P_single['p99']], [0, single_cdf[idxs_p99]], color='orange', linestyle='--', linewidth=1)
plt.plot(inference_latency_single_sorted, single_cdf, color='red', label='Single Tenant and Single Instance')
plt.plot(inference_latency_multiple_sorted, multiple_cdf, color='blue', label='Multiple Tenants and Multiple Instances')

plt.title('CDF of Inference Latencies')
plt.xlabel('Latency (ms)')
plt.ylabel('CDF (latency) [%]')
plt.legend(fontsize='small')
plt.savefig('plot/outcome/INFERENCE_CDF.pdf')
plt.close()


# Preprocess KDE
sns.kdeplot(preprocess_latency_single, color="skyblue", fill=True, label='Single Tenant and Single Instance')
sns.kdeplot(preprocess_latency_multiple, color="orange", fill=True, label='Multiple Tenants and Multiple Instances')

plt.title('KDE of Preprocessing Latencies')
plt.xlabel('Latency (ms)')
plt.ylabel('Density')

plt.legend(loc='upper right')
plt.savefig('plot/outcome/PREPROCESS_KDE.pdf')
plt.close() 

# Preprocess CDF
preprocess_latency_single_sorted = np.sort(preprocess_latency_single)
single_cdf = np.arange(1, len(preprocess_latency_single_sorted) + 1) / len(preprocess_latency_single_sorted)
idxs_p90 = np.searchsorted(preprocess_latency_single_sorted, preprocess_P_single['p90'], side='right') - 1
idxs_p99 = np.searchsorted(preprocess_latency_single_sorted, preprocess_P_single['p99'], side='right') - 1

preprocess_latency_multiple_sorted = np.sort(preprocess_latency_multiple)
multiple_cdf = np.arange(1, len(preprocess_latency_multiple_sorted) + 1) / len(preprocess_latency_multiple_sorted)
idxm_p90 = np.searchsorted(preprocess_latency_multiple_sorted, preprocess_P_multiple['p90'], side='right') - 1
idxm_p99 = np.searchsorted(preprocess_latency_multiple_sorted, preprocess_P_multiple['p99'], side='right') - 1

plt.plot([preprocess_P_multiple['p90'], preprocess_P_multiple['p90']], [0, multiple_cdf[idxm_p90]], color='cyan', linestyle='--', linewidth=1, label='P90 Latency')
plt.plot([preprocess_P_single['p90'], preprocess_P_single['p90']], [0, single_cdf[idxs_p90]], color='cyan', linestyle='--', linewidth=1)
plt.plot([preprocess_P_multiple['p99'], preprocess_P_multiple['p99']], [0, multiple_cdf[idxm_p99]], color='orange', linestyle='--', linewidth=1, label='P99 Latency')
plt.plot([preprocess_P_single['p99'], preprocess_P_single['p99']], [0, single_cdf[idxs_p99]], color='orange', linestyle='--', linewidth=1)
plt.plot(preprocess_latency_single_sorted, single_cdf, color='red', label='Single Tenant and Single Instance')
plt.plot(preprocess_latency_multiple_sorted, multiple_cdf, color='blue', label='Multiple Tenants and Multiple Instances')

plt.title('CDF of Preprocess Latencies')
plt.xlabel('Latency (ms)')
plt.ylabel('CDF (latency) [%]')
plt.legend(fontsize='small')
plt.savefig('plot/outcome/PREPROCESS_CDF.pdf')
plt.close()