import torch
import torch.nn as nn
import torch.nn.functional as F
from sdeint.euler import sdeint_euler
from sdeint.euler import odeint_euler
from torchdiffeq import odeint_adjoint as odeint
from .utils import ConcatConv2d

def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)

def downsample(x):
    return F.max_pool2d(x, 2)

def upsample(x):
    return F.max_unpool2d(x, 2)

def sde_flow(nn.Module):
    def __init__(self, n_downsample=0):
        super().__init__()
        self.n_downsample = n_downsample
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
        out = x
        for _ in range(self.n_downsample):
            out = downsample(out)
        out = self.norm1(out)
        out = self.relu1(out)
        out = self.conv1(t, out)
        out = self.norm2(out)
        out = self.relu2(out)
        out = self.conv2(t, out)
        out = self.norm3(out)
        out = self.relu3(out)
        out = self.conv3(t, out)
        out = self.norm4(out)
        for _ in range(self.n_downsample):
            out = upsample(out)
        return out


# multiplicative noise
class SDEfunc(nn.Module):
    def __init__(self, dim):
        super(SDEfunc, self).__init__()
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

    def diffusion(self, t, x, sigma):
        return sigma * x

    def dif_diffusion(self, t, x, sigma):
        return sigma

# additive noise
class SDEfunc2(nn.Module):
    def __init__(self, dim):
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
        #self.relu4 = nn.ReLU(inplace=True)
        #self.conv4 = ConcatConv2d(dim, dim, 3, 1, 1)
        #self.norm5 = norm(dim)

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

    def diffusion(self, t, x, sigma):
        return sigma

    def dif_diffusion(self, t, x, sigma):
        return 0

# dropout noise
class SDEfunc3(nn.Module):
    def __init__(self, dim, sigma):
        super(SDEfunc3, self).__init__()
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
        self.sigma = sigma
        self.out = None

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
        self.out = out
        return out

    def diffusion(self, t, x):
        return self.out * self.sigma

    def dif_diffusion(self, t, x):
        raise NotImplementedError

class SDEBlock(nn.Module):
    def __init__(self, sdefunc, mid_state):
        super(SDEBlock, self).__init__()
        self.sdefunc = sdefunc
        self.mid_state = mid_state
    
    def forward(self, x, kwargs):
        if "grid" in kwargs:
            grid = kwargs["grid"]
        else:
            grid = 0.1
        if "T" in kwargs:
            T = kwargs["T"]
        else:
            T = 1.0
        if "sigma" in kwargs:
            sigma = kwargs["sigma"]
        else:
            sigma = 0.0
        out = sdeint_euler(self.sdefunc.forward, self.sdefunc.diffusion,
            self.sdefunc.dif_diffusion, T, grid, sigma, x, self.mid_state)
        return out


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)

class SdeClassifier(nn.Module):
    def __init__(self, in_nc, noise_type="additive"):
        super(SdeClassifier, self).__init__()
        self.downsampling_layers = nn.Conv2d(in_nc, 64, 3, 1)
        if noise_type == "multiplicative":
            sde_fn = SDEfunc
        elif noise_type == "additive":
            sde_fn = SDEfunc2
        elif noise_type == "dropout":
            sde_fn = SDEfunc3
        else:
            raise ValueError
        self.sde_layer = SDEBlock(sde_fn(64), mid_state=None)
        self.pre_fc_layers = nn.Sequential(
            norm(64), 
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)), 
            Flatten()
        )
        self.fc_layer = nn.Linear(64, 10)

    def set_mid_state(self, mid_state):
        self.sde_layer.mid_state = mid_state
    
    def forward(self, x, kwargs):
        out = self.downsampling_layers(x)
        out = self.sde_layer(out, kwargs)
        out = self.pre_fc_layers(out)
        if 'logits' in kwargs and kwargs['logits']:
            out = self.fc_layer(out)
        return out

class SdeClassifier_big(nn.Module):
    def __init__(self, in_nc, sigma, mid_state, noise_type="additive", n_class=200):
        super(SdeClassifier_big, self).__init__()
        self.mid_state = mid_state
        self.downsampling_layers = [
            nn.Conv2d(in_nc, 64, 3, 1, 1),
            norm(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            norm(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
        ]
        if noise_type == "multiplicative":
            sde_fn = SDEfunc
        elif noise_type == "additive":
            sde_fn = SDEfunc2
        elif noise_type == "dropout":
            sde_fn = SDEfunc3
        else:
            raise ValueError
        self.feature_layers = [SDEBlock(sde_fn(256, sigma), self.mid_state)]
        self.fc_layers = [norm(256), nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)), Flatten(), nn.Linear(256, n_class)]
        self.model = nn.Sequential(*self.downsampling_layers, *self.feature_layers,
                *self.fc_layers)

    def set_mid_state(self, mid_state):
        self.feature_layers[0].mid_state = mid_state
   
    def forward(self, x, kwargs):
        return self.model(x, kwargs)
    
