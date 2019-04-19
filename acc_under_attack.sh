#!/bin/bash

n_ensemble=100  # 100
max_norm=0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5
sigma=0.7
test_sigma=0.7

echo sigma=${sigma}, test_sigma=${test_sigma}

CUDA_VISIBLE_DEVICES=1 python acc_under_attack.py \
    --model sde \
    --data cifar10 \
    --n_ensemble ${n_ensemble} \
    --steps 40 \
    --max_norm ${max_norm} \
    --test_sigma ${test_sigma} \
    --sigma ${sigma} #> ./results/acc_sde_cifar10_2.0.txt
