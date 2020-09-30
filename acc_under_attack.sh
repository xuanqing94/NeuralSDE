#!/bin/bash


CUDA_VISIBLE_DEVICES=3 python acc_under_attack.py \
    --n_scale 3 --batch_size 100 --step 0 \
    --sigma 1.0 \
    --n_ensemble 1 \
    --max_norm 0.0 \
    --num_avg 1 \
    --noise_type multiplicative \
    --version v2 \
    --data cifar10.1 \
