import subprocess
import psutil
import json
import time
import socket
import matplotlib.pyplot as plt

def receive_signal(host='127.0.0.1', port=12345):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen()
        conn, addr = s.accept()
        with conn:
            print(f"Connected by {addr}")
            while True:
                data = conn.recv(1024)
                if not data:
                    break
                print(f"Received {data.decode()}")

def check_if_program_is_running(program_name):
    for process in psutil.process_iter(attrs=["name", "cmdline"]):
        if program_name in process.info["name"] or \
           program_name in " ".join(process.info["cmdline"]):
            return True
    return False


# 构建Docker运行命令
docker_command = """
docker run -d --rm --name=triton --gpus=1 -p8000:8000 -p8001:8001 -p8002:8002 -v ~/model_repository_cuda12.3:/models nvcr.io/nvidia/tritonserver:24.01-py3 tritonserver --model-repository=/models
"""
# 使用subprocess运行Docker命令
try:
    subprocess.run(docker_command, shell=True, check=True)
except subprocess.CalledProcessError as e:
    print(f"Error occurred: {e}")

time.sleep(5)

iterations, cpu_util, mem_util, gpu_util, gmem_util = 0, 0, 0, 0, 0
try:
    # 首次调用cpu_percent()初始化监控
    psutil.cpu_percent(interval=None)
    time.sleep(2)
    start_cpu_usage = psutil.cpu_percent()
    start_mem_usage = psutil.virtual_memory().percent
    
    cmd = "nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv,noheader,nounits"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    gpu_utilization = result.stdout.strip().split('\n')
    gpu_utilization = [x.split(', ') for x in gpu_utilization]
    
    start_gpu, start_gmem = 0, 0
    for i, util in enumerate(gpu_utilization):
        start_gpu, start_gmem = int(util[0]), int(util[1])
    
    print("-----------------------------READY------------------------------")
    
    # 接收来自客户端的信号, 开始收集信息
    for i in range(2):
        receive_signal()
    
    cpu_util = []
    mem_util = []
    gpu_util = []
    gmem_util = []
    
    while True:
        # 再次获取CPU利用率
        cpu_cores_usage = psutil.cpu_percent(percpu=True)
        total_cpu_usage = psutil.cpu_percent()
        print(f'CPU cores usage: {cpu_cores_usage}')
        print(f'Total CPU usage: {total_cpu_usage}')
        print("--")
        
        # 获取物理内存的使用情况
        memory = psutil.virtual_memory()
        # 打印总内存、已用内存、可用内存和内存利用率
        print(f"Total memory: {memory.total / (1024**3):.2f} GB")
        print(f"Used memory: {memory.used / (1024**3):.2f} GB")
        print(f"Memory usage: {memory.percent}%")
        print("--")
        
        # 调用nvidia-smi命令获取GPU状态的JSON格式输出
        cmd = "nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv,noheader,nounits"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        # 解析输出结果
        gpu_utilization = result.stdout.strip().split('\n')
        gpu_utilization = [x.split(', ') for x in gpu_utilization]
        
        gpu, gmem = 0, 0
        for i, util in enumerate(gpu_utilization):
            print(f'GPU {i} Utilization: GPU {util[0]}%, Memory {util[1]}%')
            gpu, gmem = int(util[0]), int(util[1])
        print("--")
        
        program_name = "client.py"
        if check_if_program_is_running(program_name) and gpu != 0:
            print(f'Client is ON')
            iterations += 1
            cpu_util.append(total_cpu_usage)
            mem_util.append(memory.percent)
            gpu_util.append(gpu)
            gmem_util.append(gmem)
        else: 
            print(f'Client is OFF')
            break
        print("-----------------------------------------------------------")
        
except KeyboardInterrupt:
    print("Stop Monitoring...")
    # 停止Docker容器
    stop_command = "docker stop triton"
    try:
        subprocess.run(stop_command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while stopping the container: {e}")
finally:
    runtime = [i for i in range(len(gpu_util) + 1)]

    with open('plot/multiple/multiple_runtime.json', 'w') as file:
        json.dump(runtime, file)
    
    print("****************************GPU******************************")
    print(f"Start GPU Utilization: {start_gpu}%")
    print(f"Average GPU Utilization: {sum(gpu_util) / iterations}%")
    gpu_util.insert(0, start_gpu)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(runtime, gpu_util)
    ax.set_title('GPU Utilization Over Time')
    ax.set_xlabel('Runtime')
    ax.set_ylabel('Utilization (%)') 
    ax.grid(True)
    ax.set_ylim(0, 100)
    plt.savefig(f"multiple/outcome/GPU_UTIL.png")
    
    with open('plot/multiple/gpu_util_multiple.json', 'w') as file:
        json.dump(gpu_util, file)
    
    print("--")
    print(f"Start GPU Memory Utilization: {start_gmem}%")
    print(f"Average GPU Memory Utilization: {sum(gmem_util) / iterations}%")
    gmem_util.insert(0, start_gmem)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(runtime, gmem_util)
    ax.set_title('GPU Memory Utilization Over Time')
    ax.set_xlabel('Runtime')
    ax.set_ylabel('Utilization (%)') 
    ax.grid(True)
    ax.set_ylim(0, 100)
    plt.savefig(f"multiple/outcome/GPUMEM_UTIL.png")
    
    print("****************************CPU******************************")
    print(f"Start CPU Utilization: {start_cpu_usage}%")
    print(f"Average CPU Utilization: {sum(cpu_util) / iterations}%")
    cpu_util.insert(0, start_cpu_usage)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(runtime, cpu_util)
    ax.set_title('CPU Utilization Over Time')
    ax.set_xlabel('Runtime')
    ax.set_ylabel('Utilization (%)') 
    ax.grid(True)
    ax.set_ylim(0, 100)
    plt.savefig(f"multiple/outcome/CPU_UTIL.png")

    with open('plot/multiple/cpu_util_multiple.json', 'w') as file:
        json.dump(cpu_util, file)
    
    print("--")
    print(f"Start Main Memory Utilization: {start_mem_usage}%")
    print(f"Average Main Memory Utilization: {sum(mem_util) / iterations}%")
    mem_util.insert(0, start_mem_usage)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(runtime, mem_util)
    ax.set_title('CPU Memory Utilization Over Time')
    ax.set_xlabel('Runtime')
    ax.set_ylabel('Utilization (%)') 
    ax.grid(True)
    ax.set_ylim(0, 100)
    plt.savefig(f"multiple/outcome/CPUMEM_UTIL.png")
    

# 停止Docker容器
stop_command = "docker stop triton"
try:
    subprocess.run(stop_command, shell=True, check=True)
except subprocess.CalledProcessError as e:
    print(f"Error occurred while stopping the container: {e}")