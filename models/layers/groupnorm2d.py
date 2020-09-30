import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn import Parameter
import torch.nn.functional as F

from .weight_noise import noise_fn


class RandGroupNorm(nn.Module):
    def __init__(
        self, num_groups, num_channels, eps=1e-5, affine=True, **rand_args,
    ):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.sigma_0 = rand_args['sigma_0']
        self.N = rand_args['N']
        self.init_s = rand_args['init_s']
        self.alpha = rand_args['alpha']
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.mu_weight = Parameter(torch.Tensor(num_channels))
            self.sigma_weight = Parameter(torch.Tensor(num_channels))
            self.register_buffer("eps_weight", torch.Tensor(num_channels))
            self.mu_bias = Parameter(torch.Tensor(num_channels))
            self.sigma_bias = Parameter(torch.Tensor(num_channels))
            self.register_buffer("eps_bias", torch.Tensor(num_channels))
        else:
            self.register_parameter("mu_weight", None)
            self.register_parameter("sigma_weight", None)
            self.register_parameter("eps_weight", None)
            self.register_parameter("mu_bias", None)
            self.register_parameter("sigma_bias", None)
            self.register_parameter("eps_bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            init.ones_(self.mu_weight)
            self.sigma_weight.data.fill_(self.init_s)
            init.zeros_(self.mu_bias)
            self.sigma_bias.data.fill_(self.init_s)

    def forward(self, input):
        if self.affine:
            weight = noise_fn(
                self.mu_weight,
                self.sigma_weight,
                self.eps_weight,
                self.sigma_0,
                self.N,
                self.alpha,
            )
            bias = noise_fn(
                self.mu_bias,
                self.sigma_bias,
                self.eps_bias,
                self.sigma_0,
                self.N,
                self.alpha,
            )
        else:
            weight = None
            bias = None
        return F.group_norm(input, self.num_groups, weight, bias, self.eps)

    def extra_repr(self):
        return "{num_groups}, {num_channels}, eps={eps}, " "affine={affine}".format(
            **self.__dict__
        )
