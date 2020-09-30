#!/bin/bash

device=0,1
batch_size=256
noise_type=dropout
data=tiny-imagenet
n_ensemble=1

for levels in 1 2 3 4 5
do
	for sigma in 0.9
	do
		test_sigma=0
		echo "nonadv testing" "sigma =" ${sigma} "test_sigma =" ${test_sigma}
		echo "level = " ${levels}
		ckpt_f=./ckpt/sde_${data}_${sigma}_${noise_type}.pth
		CUDA_VISIBLE_DEVICES=${device} python nonadv_evaluation.py \
			--data $data \
			--batch_size $batch_size \
			--levels $levels \
			--ckpt $ckpt_f \
			--sigma $sigma \
			--test_sigma $test_sigma \
			--noise_type $noise_type \
			--n_ensemble $n_ensemble \
			> >(tee ./results/nonadv_acc/nonadv_acc_sde_${data}_train\=${sigma}_test\=${sigma}_${noise_type}.txt) 
		#$device=$((device+1))
	done
done
