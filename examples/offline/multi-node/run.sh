#!/bin/sh$


CUDA_LAUNCH_BLOCKING=1 deepspeed \
    --master_port 42000 \
    --hostfile ./hostfile \
    main.py \
    --deepspeed_config "deepspeed_config.json" 
