import torch
import torch.nn as nn

from .layers.conv2d import RandConv2d
from .layers.concat_conv2d import ConcatConv2d, RandConcatConv2d
from .layers.concat_conv2d import ConcatConv2d_v2, RandConcatConv2d_v2
from .layers.groupnorm2d import RandGroupNorm

def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)

def rand_norm(dim, **rand_args):
    return RandGroupNorm(min(32, dim), dim, **rand_args)

class FlowFn(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = norm(dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm2 = norm(dim)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm3 = norm(dim)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv3 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm4 = norm(dim)

    def forward(self, t, x):
        out = self.norm1(x)
        out = self.relu1(out)
        out = self.conv1(t, out)
        out = self.norm2(out)
        out = self.relu2(out)
        out = self.conv2(t, out)
        out = self.norm3(out)
        out = self.relu3(out)
        out = self.conv3(t, out)
        out = self.norm4(out)
        return out

class FlowFn_v2(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = norm(dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = ConcatConv2d_v2(dim, dim, 3, 1, 1)
        self.norm2 = norm(dim)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim, dim, 3, 1, 1)
        self.norm3 = norm(dim)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(dim, dim, 3, 1, 1)
        self.norm4 = norm(dim)

    def forward(self, t, x):
        out = self.norm1(x)
        out = self.relu1(out)
        out = self.conv1(t, out)
        out = self.norm2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        out = self.norm3(out)
        out = self.relu3(out)
        out = self.conv3(out)
        out = self.norm4(out)
        return out

class RandFlowFn(nn.Module):
    def __init__(self, dim, **rand_args):
        super().__init__()
        self.norm1 = rand_norm(dim, **rand_args)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = RandConcatConv2d(dim, dim, 3, 1, 1, **rand_args)
        self.norm2 = rand_norm(dim, **rand_args)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = RandConcatConv2d(dim, dim, 3, 1, 1, **rand_args)
        self.norm3 = rand_norm(dim, **rand_args)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv3 = RandConcatConv2d(dim, dim, 3, 1, 1, **rand_args)
        self.norm4 = rand_norm(dim, **rand_args)

    def forward(self, t, x):
        out = self.norm1(x)
        out = self.relu1(out)
        out = self.conv1(t, out)
        out = self.norm2(out)
        out = self.relu2(out)
        out = self.conv2(t, out)
        out = self.norm3(out)
        out = self.relu3(out)
        out = self.conv3(t, out)
        out = self.norm4(out)
        return out
        
class RandFlowFn_v2(nn.Module):
    def __init__(self, dim, **rand_args):
        super().__init__()
        self.norm1 = rand_norm(dim, **rand_args)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = RandConcatConv2d_v2(dim, dim, 3, 1, 1, **rand_args)
        self.norm2 = rand_norm(dim, **rand_args)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = RandConv2d(dim, dim, 3, 1, 1, **rand_args)
        self.norm3 = rand_norm(dim, **rand_args)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv3 = RandConv2d(dim, dim, 3, 1, 1, **rand_args)
        self.norm4 = rand_norm(dim, **rand_args)

    def forward(self, t, x):
        out = self.norm1(x)
        out = self.relu1(out)
        out = self.conv1(t, out)
        out = self.norm2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        out = self.norm3(out)
        out = self.relu3(out)
        out = self.conv3(out)
        out = self.norm4(out)
        return out
