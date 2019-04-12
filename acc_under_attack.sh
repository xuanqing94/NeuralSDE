#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python acc_under_attack.py \
    --model sde \
    --data cifar10 \
    --n_ensemble 20 \
    --steps 40 \
    --max_norm 0,0.1,0.2,0.3,0.4 \
    --sigma 20 #> ./results/acc_sde_cifar10_2.0.txt
