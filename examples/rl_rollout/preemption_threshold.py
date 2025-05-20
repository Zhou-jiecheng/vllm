#!/usr/bin/env python3
import argparse
import requests
import time
import random
import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import uuid
import numpy as np

VLLM_API_URL = "http://localhost:30913/v1/chat/completions"
MODEL_NAME = "/cpfs01/user/zhoujiecheng/vllm/examples/rl_rollout/models/Qwen2.5-14B"


class RequestConfig(BaseModel):
    prompt_len: int
    output_len: int
    request_id: str = ""
    input_ids: List[int] = []
    sampling_params: Dict[str, Any] = dict()
    ignore_eos: bool

def generate_random_inputs(length: int) -> list:
    """生成指定长度的随机token id列表"""
    # 大多数模型的词汇表大小在20000-50000之间，我们用30000作为上限
    return [random.randint(0, 30000) for _ in range(length)]

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

def send_request(idx=0) -> None:
    """发送随机长度的请求并打印响应时间"""
    # 随机生成输入和输出长度
    input_lengths = list(range(100, 2001, 50))
    output_lengths = list(range(7300, 12001, 200))

    input_length = input_lengths[9]
    output_length = output_lengths[idx]

    # 生成随机输入token ids
    input_ids = generate_random_inputs(input_length)
    
    # 创建请求
    request_data = RequestConfig(
                    prompt_len=input_length,
                    output_len=output_length,
                    request_id=str(uuid.uuid4()),
                    input_ids=input_ids,
                    sampling_params={
                        "temperature": 0.9,
                        "top_p": 1,
                        "n": 1,
                        "use_beam_search": False,
                        "n": 1,
                    },
                    ignore_eos=True
                )
    
    payload = request_config_to_vllm_payload(request_data)


    # 打印请求信息
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] 发送请求:")
    print(f"  - 输入长度: {input_length} tokens")
    print(f"  - 输出长度: {output_length} tokens")
    
    # 发送请求并计时
    start_time = time.time()
    response = send_vllm_request(payload)
    elapsed_time = time.time() - start_time
    
    # 检查响应     
    print(f"  √ 请求成功! 耗时: {elapsed_time:.2f}秒")
    return response,elapsed_time

def main():
    parser = argparse.ArgumentParser(description="定期向vLLM服务器发送随机长度的请求")
    parser.add_argument("--interval", type=float, default=1.0, 
                        help="请求间隔(秒)")
    parser.add_argument("--count", type=int, default=10, 
                        help="请求次数，-1表示无限")
    
    args = parser.parse_args()

    print(f"配置: 输入长度=100-2000, 输出长度=2000-30000, 间隔={args.interval}秒")
    
    count = 0
    elapsed_times = []
    try:
        while args.count == -1 or count < args.count:
            
            res,elapsed_time = send_request(count)
            elapsed_times.append(elapsed_time)
            count += 1
            # 如果已达到请求数限制，退出循环
            if args.count != -1 and count >= args.count:
                break
                
            # 等待指定的间隔时间
            print(f"等待 {args.interval} 秒后发送下一请求...")
            time.sleep(args.interval)
            
    except KeyboardInterrupt:
        print("\n用户中断，停止发送请求...")
    
    print(f"\n总共发送了 {count} 个请求")
    print(f"time spending:")
    print(elapsed_times)

if __name__ == "__main__":
    RANDOM_SEED = 42  # 可以设置为任意整数
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    main()