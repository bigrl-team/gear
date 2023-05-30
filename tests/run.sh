#!/bin/sh$


CUDA_LAUNCH_BLOCKING=1 deepspeed \
    --master_port 42000 \
    --include "localhost:0,1" \
    --hostfile ./hostfile \
    test_offline_loader.py \
    --deepspeed_config "deepspeed_config.json" \
    --cf "config.json" \
    --output_dir "./logs"
