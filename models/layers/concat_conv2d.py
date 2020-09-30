import math
import torch
import torch.nn as nn

from .conv2d import RandConv2d

import pdb

def time_encoding(t, shape):
    batch_size, embedding_dim, H, W = shape
    half_dim = embedding_dim // 2
    emb = math.log(100) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float).cuda() * -emb)
    emb = emb * t
    emb = torch.cat([torch.sin(emb), torch.cos(emb)])
    emb = emb.view(1, embedding_dim, 1, 1).expand(batch_size, -1, H, W)
    return emb

class ConcatConv2d(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        ksize=3,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        transpose=False,
    ):
        super(ConcatConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in,
            dim_out,
            kernel_size=ksize,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1, :, :]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)

class ConcatConv2d_v2(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        ksize=3,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        transpose=False,
    ):
        super().__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self.embedding_dim = 64
        self._layer = module(
            dim_in + self.embedding_dim,
            dim_out,
            kernel_size=ksize,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, t, x):
        batch_size, _, H, W = x.size()
        tt = time_encoding(t, (batch_size, self.embedding_dim, H, W))
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)

class RandConcatConv2d(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        ksize=3,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        **rand_args,
    ):
        super().__init__()
        self._layer = RandConv2d(
            dim_in + 1,
            dim_out,
            ksize,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            **rand_args,
        )

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1, :, :]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)


class RandConcatConv2d_v2(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        ksize=3,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        **rand_args,
    ):
        super().__init__()
        self.embedding_dim = 64
        self._layer = RandConv2d(
            dim_in + self.embedding_dim,
            dim_out,
            ksize,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            **rand_args,
        )

    def forward(self, t, x):
        batch_size, _, H, W = x.size()
        tt = time_encoding(t, (batch_size, self.embedding_dim, H, W))
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)
