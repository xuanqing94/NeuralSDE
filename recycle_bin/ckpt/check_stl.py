#!/usr/bin/env python

import os
import shutil
import torch


for f in os.listdir('.'):
    if 'stl10' in f:
        model = torch.load(f)
        if model['model.12.bias'].numel() != 10:
            shutil.move(f, f"../ckpt_incorrect/{f}")
            print(f"Move file {f}")
