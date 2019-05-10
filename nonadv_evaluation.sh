#!/bin/bash

device=0
batch_size=128
noise_type=dropout
data=cifar10
n_ensemble=100

for sigma in 0.6 0.7 0.8 0.9
do
	echo "nonadv testing" "sigma =" ${sigma} 
	CUDA_VISIBLE_DEVICES=${device} python nonadv_evaluation.py \
		--data $data \
		--batch_size $batch_size \
		--sigma $sigma \
		--test_sigma $sigma \
		--noise_type $noise_type \
		--n_ensemble $n_ensemble \
		> >(tee ./results/nonadv_acc/nonadv_acc_sde_${data}_train\=${sigma}_test\=${sigma}_${noise_type}.txt) &
	device=$((device+1))
done
