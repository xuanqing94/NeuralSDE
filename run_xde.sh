#!/bin/bash

#./sde_classifier.py --data cifar10 --sigma 0.5 > ./log_sde_cifar10_0.5.txt

data=cifar10
sigma=0.7

echo Training with sigma=${sigma}

CUDA_VISIBLE_DEVICES=4 python ./sde_classifier.py \
	--data $data \
	--sigma $sigma \
	> >(tee ./log/log_sde_${data}_${sigma}.txt) 2>./log/error.log
