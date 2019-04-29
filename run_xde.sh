#!/bin/bash

#./sde_classifier.py --data cifar10 --sigma 0.5 > ./log_sde_cifar10_0.5.txt

device=0
for s in 1.4 1.6 1.8
do
	data=cifar10
	sigma=$s
	noise_type=multiplicative
	#epochs=20,20,10
	epochs=40,40,20,20
	echo Training with sigma=${sigma}
	
	CUDA_VISIBLE_DEVICES=$device python ./sde_classifier.py \
		--data $data \
		--sigma $sigma \
		--epochs $epochs \
		--noise_type $noise_type \
		> >(tee ./log/log_sde_${data}_${sigma}_${noise_type}.txt) &
	device=$((device+1))
done
