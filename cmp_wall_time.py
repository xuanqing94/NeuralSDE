#!/usr/bin/env python
import time
import torch
import torch.nn as nn


from torchdiffeq import odeint_adjoint as odeint
from sdeint.euler import sdeint_euler

""" Compare the wall clock time, as requested by Reviewer 3. """


class ConcatConv2d(nn.Module):
    """ Concat with time and then do the convolution. """
    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in + 1, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )
    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1, :, :]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)

def norm(dim):
    """ Group normalization helper function. """
    return nn.GroupNorm(min(32, dim), dim)

class ODEfunc(nn.Module):
    def __init__(self, dim):
        super(ODEfunc, self).__init__()
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

# multiplicative noise, copied from Neural-SDE
class SDEfunc(ODEfunc):
    def __init__(self, dim, sigma):
        super(SDEfunc, self).__init__(dim)
        self.sigma = sigma

    def diffusion(self, t, x):
        return self.sigma * x

    def dif_diffusion(self, t, x):
        return self.sigma



class ODEBlock(nn.Module):
    """ Encapsule the integration block """
    def __init__(self, odefunc, grid=0.1):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 1]).float()
        self.grid = grid

    def forward(self, x):
        out = odeint(self.odefunc, x, self.integration_time, method='euler', options={'step_size': self.grid})
        return out[1]

class SDEBlock(nn.Module):
    def __init__(self, sdefunc, grid=0.1):
        super(SDEBlock, self).__init__()
        self.sdefunc = sdefunc
        self.T = 1.0
        self.grid = grid


    def forward(self, x):
        out = sdeint_euler(self.sdefunc.forward, self.sdefunc.diffusion,
            self.sdefunc.dif_diffusion, self.T, self.grid, x)
        return out

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)

class SdeClassifier(nn.Module):
    def __init__(self, in_nc, sigma, grid=0.1):
        super(SdeClassifier, self).__init__()
        self.downsampling_layers = [
            nn.Conv2d(in_nc, 64, 3, 1),
        ]
        sde_fn = SDEfunc
        self.feature_layers = [SDEBlock(sde_fn(64, sigma), grid)]
        self.fc_layers = [norm(64), nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)), Flatten(), nn.Linear(64, 10)]
        self.model = nn.Sequential(*self.downsampling_layers, *self.feature_layers,
                *self.fc_layers)

    def forward(self, x):
        return self.model(x)


class OdeClassifier(nn.Module):
    def __init__(self, in_nc, grid=0.1):
        super(OdeClassifier, self).__init__()
        self.downsampling_layers = [
                nn.Conv2d(in_nc, 64, 3, 1),
        ]
        ode_fn = ODEfunc
        self.feature_layers = [ODEBlock(ode_fn(64))]
        self.fc_layers = [norm(64), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)), Flatten(), nn.Linear(64, 10)]
        self.model = nn.Sequential(*self.downsampling_layers, *self.feature_layers, *self.fc_layers)

    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    grid = 5.0e-2
    #model = SdeClassifier(in_nc=3, sigma=0.1, grid=grid)
    model = OdeClassifier(in_nc=3, grid=grid)
    model.cuda()
    loss_fn = nn.CrossEntropyLoss()
    n_iter = 100
    beg = time.time()
    for _ in range(n_iter):
        input_batch = torch.rand(64, 3, 32, 32).cuda()
        output_batch = torch.randint(0, 10, (64,), dtype=torch.long).cuda()
        model.zero_grad()
        pred = model(input_batch)
        loss = loss_fn(pred, output_batch)
        loss.backward()
    end = time.time()
    print(f'Round trip time: {(end - beg) / n_iter}.')
