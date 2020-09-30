#!/bin/bash

data=tiny-imagenet

CUDA_VISIBLE_DEVICES=1 python ./sde_classifier.py \
	--data $data \
	--sigma 0.0 \
	--epochs 0 \
	--noise_type dropout \
	--model_size
