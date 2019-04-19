#!/bin/bash

data=cifar10
sigma=0.7
test_sigma=0
step_size=0.02
n_ensemble=2000

echo sigma=${sigma}, test_sigma=${test_sigma}

CUDA_VISIBLE_DEVICES=0 python visualize_mid_state.py \
	--data ${data} \
	--sigma ${sigma} \
	--test_sigma ${test_sigma} \
	--step_size ${step_size} \
	--n_ensemble ${n_ensemble} \
	> >(tee ./results/snr/snr_sde_train\=${sigma}_test\=${test_sigma}.txt)
