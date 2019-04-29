#!/bin/bash

data=cifar10
dims=64,64,64
strides=1,1,1,1
num_blocks=2
batch_size=512
img_dir=./img
lr=1.0e-3

CUDA_VISIBLE_DEVICES=0,1,2,3 python main_flow.py \
	--data $data \
	--dims $dims \
	--strides $strides \
	--num_blocks $num_blocks \
	--batch_size $batch_size \
	--img_dir $img_dir \
	--lr $lr \
	#> >(tee ./log/log_train_flow.txt) 2>./log/error.log
