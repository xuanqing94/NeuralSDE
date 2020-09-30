#!/bin/bash

data=cifar10
sigma=2.0
test_sigma=2.0
step_size=0.02
n_ensemble=2000
noise_type=dropout

echo sigma=${sigma}, test_sigma=${test_sigma}

CUDA_VISIBLE_DEVICES=5 python visualize_mid_state.py \
	--data ${data} \
	--sigma ${sigma} \
	--test_sigma ${test_sigma} \
	--step_size ${step_size} \
	--n_ensemble ${n_ensemble} \
	--noise_type ${noise_type} \
	> >(tee ./results/snr/snr_sde_train\=${sigma}_test\=${test_sigma}_${noise_type}.txt)
