#!/bin/bash

# CIFAR: 83.3

for data in stl10 tiny-imagenet
do
	CUDA_VISIBLE_DEVICES=4,5 python resnet_classifier.py --data $data
done
