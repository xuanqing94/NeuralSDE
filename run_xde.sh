#!/bin/bash

#./sde_classifier.py --data cifar10 --sigma 0.5 > ./log_sde_cifar10_0.5.txt

data=cifar10
sigma=20

CUDA_VISIBLE_DEVICES=1 python ./sde_classifier.py \
	--data $data \
	--sigma $sigma \
	> >(tee log_sde_${data}_${sigma}.txt) 2>./error.log
