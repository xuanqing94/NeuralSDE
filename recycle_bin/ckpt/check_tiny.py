#!/usr/bin/env python

import os
import shutil
import torch


for f in os.listdir('.'):
    if 'tiny-imagenet' in f:
        model = torch.load(f)
        #print(f)
        #print(model.keys())
        #continue
        if model['model.12.bias'].numel() != 200:
            #shutil.move(f, f"../ckpt_incorrect/{f}")
            print(f"Move file {f}")
