#!/bin/bash

#./sde_classifier.py --data cifar10 --sigma 0.5 > ./log_sde_cifar10_0.5.txt

device=5
data=cifar10
sigma=0.0
noise_type=additive
epochs=60,40,40,20 # for cifar10
#epochs=30,30,20,20 # for stl10
echo Training with sigma=${sigma}
CUDA_VISIBLE_DEVICES=$device python ./sde_classifier.py \
    --data $data \
    --sigma $sigma \
    --grid_size 0.1 \
    --epochs $epochs \
    --noise_type $noise_type
