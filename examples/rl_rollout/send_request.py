import random
import string
import datetime 
import uuid
import numpy as np
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# 配置
VLLM_API_URL = "http://localhost:30913/v1/chat/completions"
MODEL_NAME = "/cpfs01/user/zhoujiecheng/vllm/examples/rl_rollout/models/Qwen2.5-14B"
BATCH_SIZE = 128        # 修改为64个请求
MAX_WORKERS = 32       # 最大并发数

# prompt_len 的正态分布参数
INPUT_LEN_MEAN = 470   # 均值
INPUT_LEN_MIN = 120    # 最小值
INPUT_LEN_MAX = 1400   # 最大值
INPUT_LEN_STD = 200    # 标准差

# 词汇表参数
VOCAB_SIZE = 150000     # 词汇表大小
SPECIAL_TOKENS = {     # 特殊token
    'bos_token': 1,    # 开始标记
    'eos_token': 2,    # 结束标记
    'pad_token': 0,    # 填充标记
}

def random_input_ids(length, vocab_size=VOCAB_SIZE):
    """生成随机的input_ids
    
    Args:
        length: 序列长度
        vocab_size: 词汇表大小
    
    Returns:
        List[int]: 随机生成的token ids列表
    """
    # 确保序列以bos_token开始
    input_ids = [SPECIAL_TOKENS['bos_token']]
    
    # 生成中间的随机token ids
    # 避免使用特殊token和填充token
    valid_tokens = list(range(3, vocab_size))
    input_ids.extend(random.choices(valid_tokens, k=length-1))
    
    return input_ids

# 读取workload数据
def load_workload_data():
    with open('./output_dsitribution/rollout_workload_deepcoder.txt', 'r') as f:
        return [int(line.strip()) for line in f if line.strip()]

# 将workload数据划分为区间
def create_workload_bins(workload_data, num_bins=20):
    # 计算分位数
    percentiles = np.percentile(workload_data, np.linspace(0, 100, num_bins + 1))
    
    # 创建区间
    bins = []
    for i in range(num_bins):
        lower = int(percentiles[i])
        upper = int(percentiles[i + 1])
        # 获取该区间内的所有值
        values = [x for x in workload_data if lower <= x <= upper]
        if values:  # 确保区间内有值
            bins.append(values)
    
    return bins

class RequestConfig(BaseModel):
    prompt_len: int
    output_len: int
    request_id: str = ""
    input_ids: List[int] = []
    sampling_params: Dict[str, Any] = dict()
    ignore_eos: bool

def generate_normal_distributed_length():
    """生成符合正态分布的长度，并确保在指定范围内"""
    while True:
        length = int(np.random.normal(INPUT_LEN_MEAN, INPUT_LEN_STD))
        if INPUT_LEN_MIN <= length <= INPUT_LEN_MAX:
            return length

def generate_workload_distributed_lengths(workload_bins, num_requests):
    """生成指定数量的输出长度，确保每个区间都被采样"""
    num_bins = len(workload_bins)
    # 计算每个区间应该生成的请求数
    base_per_bin = num_requests // num_bins
    remainder = num_requests % num_bins
    
    # 为每个区间分配请求数
    requests_per_bin = [base_per_bin] * num_bins
    # 将余数随机分配给一些区间
    for i in range(remainder):
        requests_per_bin[random.randint(0, num_bins-1)] += 1
    
    # 从每个区间生成指定数量的值
    output_lengths = []
    for bin_idx, num_requests in enumerate(requests_per_bin):
        bin_values = workload_bins[bin_idx]
        # 如果区间内的值不够，允许重复采样
        if len(bin_values) < num_requests:
            output_lengths.extend(random.choices(bin_values, k=num_requests))
        else:
            output_lengths.extend(random.sample(bin_values, num_requests))
    
    # 打乱顺序
    random.shuffle(output_lengths)
    return output_lengths

def gen_dummy_requests(n, workload_bins):
    """生成n个请求配置，每8个request使用相同的input_ids"""
    # 生成输出长度
    output_lengths = generate_workload_distributed_lengths(workload_bins, n)
    
    # 生成请求配置
    requests = []
    for i in range(0, n, 8):
        # 每8个request生成一次input_ids
        prompt_len = generate_normal_distributed_length()
        input_ids = random_input_ids(prompt_len)
        
        # 使用相同的input_ids生成8个request
        for j in range(8):
            if i + j < n:  # 确保不超出总请求数
                requests.append(RequestConfig(
                    prompt_len=prompt_len,
                    output_len=output_lengths[i + j],
                    request_id=str(uuid.uuid4()),
                    input_ids=input_ids,
                    sampling_params={
                        "temperature": 0.9,
                        "top_p": 1,
                        "do_sample": False
                    },
                    ignore_eos=True
                ))
    
    return requests

def request_config_to_vllm_payload(config: RequestConfig):
    prompt_tokens = [str(x) for x in config.input_ids]
    prompt_text = " ".join(prompt_tokens)
    messages = [
        {"role": "user", "content": f"{prompt_text}"}
    ]
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": config.output_len,
        "ignore_eos": True,
        **config.sampling_params
    }
    return payload

def send_vllm_request(payload):
    headers = None
    try:
        resp = requests.post(VLLM_API_URL, json=payload, headers=headers, timeout=1500)
        if resp.ok:
            return resp.json()
        else:
            return {"error": resp.status_code, "detail": resp.text}
    except Exception as e:
        return {"error": "exception", "detail": str(e)}

if __name__ == "__main__":
    # 加载workload数据并创建区间
    RANDOM_SEED = 42  # 可以设置为任意整数
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    workload_data = load_workload_data()
    workload_bins = create_workload_bins(workload_data, num_bins=16)
    
    # 生成请求
    dummy_requests = gen_dummy_requests(BATCH_SIZE, workload_bins)
    payloads = [request_config_to_vllm_payload(r) for r in dummy_requests]

    # 打印请求统计信息
    prompt_lengths = [r.prompt_len for r in dummy_requests]
    output_lengths = [r.output_len for r in dummy_requests]
    
    print(f"请求统计信息:")
    print(f"总请求数: {BATCH_SIZE}")
    
    print(f"\n输入长度统计:")
    print(f"  最小值: {min(prompt_lengths)}")
    print(f"  最大值: {max(prompt_lengths)}")
    print(f"  平均值: {sum(prompt_lengths)/len(prompt_lengths):.2f}")
    print(f"  标准差: {np.std(prompt_lengths):.2f}")
    
    print(f"\n输出长度统计:")
    print(f"  最小值: {min(output_lengths)}")
    print(f"  最大值: {max(output_lengths)}")
    print(f"  平均值: {sum(output_lengths)/len(output_lengths):.2f}")
    print(f"  标准差: {np.std(output_lengths):.2f}")
    print(f"  中位数: {np.median(output_lengths):.2f}")
    print(f"  90分位数: {np.percentile(output_lengths, 90):.2f}")
    
    # 打印区间分布信息
    print(f"\n输出长度区间分布:")
    for i, bin_values in enumerate(workload_bins):
        # 统计当前区间在生成的请求中的数量
        bin_count = sum(1 for x in output_lengths if min(bin_values) <= x <= max(bin_values))
        print(f"  区间 {i+1}: {min(bin_values)} - {max(bin_values)}, 原始样本数: {len(bin_values)}, 生成样本数: {bin_count}")
    
    print(f"\n开始发送请求到 {VLLM_API_URL} ...\n")

    # 发送请求
    start_time = datetime.datetime.now()
    results = []
    with ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
        future_to_idx = {executor.submit(send_vllm_request, p): i for i, p in enumerate(payloads)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                res = future.result()
                results.append((idx, res))
            except Exception as exc:
                results.append((idx, {"error": "exception", "detail": str(exc)}))
    end_time = datetime.datetime.now()
    print(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总耗时: {end_time - start_time}")