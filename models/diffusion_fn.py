# Different noise types
# TODO implement jump-duffusion noises

import torch
import torch.nn as nn


class MultiplicativeNoise(nn.Module):
    def __init__(self, sigma):
        super().__init__()
        self.register_buffer("sigma", torch.tensor(sigma, dtype=torch.float))

    def diffusion(self, t, x):
        return self.sigma * x

    def dif_diffusion(self, t, x):
        return self.sigma


class AdditiveNoise(nn.Module):
    def __init__(self, sigma):
        super().__init__()
        self.register_buffer("sigma", torch.tensor(sigma, dtype=torch.float))

    def diffusion(self, t, x):
        return self.sigma

    def dif_diffusion(self, t, x):
        return 0


class NoNoise(nn.Module):
    def __init__(self, sigma):
        super().__init__()
    
    def diffusion(self, t, x):
        return 0.0

    def dif_diffusion(self, t, x):
        return 0.0
