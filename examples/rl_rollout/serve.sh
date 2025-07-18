#!/bin/bash

export LOGLEVEL=DEBUG
LOG_DIR="/cpfs01/user/zhoujiecheng/vllm/examples/rl_rollout/logs"

# 创建时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# 创建带时间戳的日志目录
LOG_FILE="${LOG_DIR}/${TIMESTAMP}.log"
# export VLLM_TEST_ARTIFICIAL_PREEMPT=False 
# export PREEMPTION_LENGTH=7000

python -m vllm.entrypoints.openai.api_server \
    --model /cpfs01/user/zhoujiecheng/vllm/examples/rl_rollout/models/Qwen2.5-14B \
    --load-format dummy \
    --port 30913 \
    --max-num-batched-tokens 8192 \
    --tensor-parallel-size 2 \
    --max-model-len 40000 \
    --disable-log-requests \
    --gpu-memory-utilization 0.8 \
    --preemption-mode recompute \
    --seed 42 \
    --distributed-executor-backend ray 2>&1 | tee ${LOG_FILE}