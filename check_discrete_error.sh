#!/bin/bash

CUDA_VISIBLE_DEVICES=5 python check_discrete_error.py \
    --data cifar10 \
    --n_ensemble 100 \
    --grid_size 0.1,0.01,0.001 \
    --T 1.0 \
    --noise_type multiplicative \
    --test_sigma 10 \
