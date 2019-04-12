import torch
import torch.nn as nn
from sdeint.euler import sdeint_euler
from sdeint.euler import odeint_euler
from torchdiffeq import odeint_adjoint as odeint

def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)

class ConcatConv2d(nn.Module):
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

# multiplicative noise
class SDEfunc(nn.Module):
    def __init__(self, dim, sigma):
        super(SDEfunc, self).__init__()
        self.norm1 = norm(dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm2 = norm(dim)
        self.conv2 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm3 = norm(dim)
        self.sigma = sigma

    def forward(self, t, x):
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(t, out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(t, out)
        out = self.norm3(out)
        return out

    def diffusion(self, t, x):
        return self.sigma * x

    def dif_diffusion(self, t, x):
        return self.sigma

# additive noise
class SDEfunc2(nn.Module):
    def __init__(self, dim, sigma):
        super(SDEfunc2, self).__init__()
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
        self.relu4 = nn.ReLU(inplace=True)
        self.conv4 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm5 = norm(dim)
        self.sigma = sigma

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
        #out = self.relu4(out)
        #out = self.conv4(t, out)
        #out = self.norm5(out)
        return out

    def diffusion(self, t, x):
        return self.sigma

    def dif_diffusion(self, t, x):
        return 0

class SDEBlock(nn.Module):
    def __init__(self, sdefunc, mid_state, grid=0.1):
        super(SDEBlock, self).__init__()
        self.sdefunc = sdefunc
        self.T = 1.0
        self.grid = grid
        self.mid_state = mid_state


    def forward(self, x):
        out = sdeint_euler(self.sdefunc.forward, self.sdefunc.diffusion,
            self.sdefunc.dif_diffusion, self.T, self.grid, x, self.mid_state)
        return out


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)

class SdeClassifier(nn.Module):
    def __init__(self, in_nc, sigma, mid_state, grid=0.1):
        super(SdeClassifier, self).__init__()
        self.mid_state = mid_state
        self.downsampling_layers = [
            nn.Conv2d(in_nc, 64, 3, 1),
            #norm(64),
            #nn.ReLU(inplace=True),
            #nn.Conv2d(64, 64, 4, 2, 1),
            #norm(64),
            #nn.ReLU(inplace=True),
            #nn.Conv2d(64, 64, 4, 2, 1),
        ]
        self.feature_layers = [SDEBlock(SDEfunc2(64, sigma), self.mid_state, grid)]
        self.fc_layers = [norm(64), nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)), Flatten(), nn.Linear(64, 10)]
        self.model = nn.Sequential(*self.downsampling_layers, *self.feature_layers,
                *self.fc_layers)

    def set_mid_state(self, mid_state):
        self.feature_layers[0].mid_state = mid_state
   
    def forward(self, x):
        return self.model(x)

