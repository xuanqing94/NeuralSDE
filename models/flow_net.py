"""Implements multi-scale flow based classifier."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils import ConcatConv2d


def downsample(x, scale):
    if scale == 1:
        return x
    return F.avg_pool2d(x, kernel_size=scale, stride=scale)


def upsample(x, scale):
    if scale == 1:
        return x
    return F.interpolate(x, scale_factor=scale, mode="nearest")


class MultiScaleFlow(nn.Module):
    def __init__(self, flows):
        super().__init__()
        self.flows = nn.ModuleList(flows)

    def forward(self, x):
        outputs = []
        for k, flow in enumerate(self.flows):
            out = flow(downsample(x, 2 ** k))
            outputs.append(out)
        output = outputs[0]
        for i in range(1, len(outputs)):
            output += upsample(outputs[i], 2 ** i)
        return output
