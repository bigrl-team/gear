#!/bin/sh$


deepspeed \
    --master_port 42000 \
    --include "localhost:0" \
    --hostfile ./hostfile \
    main.py \
    --deepspeed_config deepspeed_config.json
