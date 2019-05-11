#!/bin/bash

n_ensemble=100  # 100
#max_norm=0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5   # for cifar10
max_norm=0.0                                              # for debugging
#max_norm=0.0,0.35,0.75,1.0,1.35,1.75,2.0,2.35,2.75,3.0   # for mnist
noise_type=multiplicative
data=cifar10
device=5

for sig in 1.2
do
	sigma=$sig
	test_sigma=$sig
	echo sigma=${sigma}, test_sigma=${test_sigma}, noise_type=${noise_type}
	CUDA_VISIBLE_DEVICES=${device} python acc_under_attack.py \
	    --model sde \
	    --data $data \
	    --n_ensemble ${n_ensemble} \
	    --steps 40 \
	    --num_avg 1 \
	    --max_norm ${max_norm} \
	    --test_sigma ${test_sigma} \
	    --sigma ${sigma} \
	    --noise_type ${noise_type} \
	    > >(tee ./results/acc/acc_sde_${data}_train\=${sigma}_test\=${test_sigma}_${noise_type}.txt) 
	device=$((device+1))
done
