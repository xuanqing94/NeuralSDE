import math
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from .weight_noise import noise_fn


class RandLinear(nn.Module):
    def __init__(
        self, in_features, out_features, bias=True, **rand_args,
    ):
        """Linear layer with random weights.
        Args:
            in_features: size of input dimension.
            out_features: size of output dimension.
            bias: whether bias is added or not.
            rand_args: N, sigma_0, init_s, alpha.
        """
        super(RandLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_0 = rand_args['sigma_0']
        self.N = rand_args['N']
        self.init_s = rand_args['init_s']
        self.alpha = rand_args['alpha']
        self.mu_weight = Parameter(torch.Tensor(out_features, in_features))
        self.sigma_weight = Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer("eps_weight", torch.Tensor(out_features, in_features))
        if bias:
            self.mu_bias = Parameter(torch.Tensor(out_features))
            self.sigma_bias = Parameter(torch.Tensor(out_features))
            self.register_buffer("eps_bias", torch.Tensor(out_features))
        else:
            self.register_parameter("mu_bias", None)
            self.register_parameter("sigma_bias", None)
            self.register_buffer("eps_bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.mu_weight.size(1))
        self.mu_weight.data.uniform_(-stdv, stdv)
        self.sigma_weight.data.fill_(self.init_s)
        self.eps_weight.data.zero_()
        if self.mu_bias is not None:
            self.mu_bias.data.uniform_(-stdv, stdv)
            self.sigma_bias.data.fill_(self.init_s)
            self.eps_bias.data.zero_()

    def forward(self, input):
        weight = noise_fn(
            self.mu_weight,
            self.sigma_weight,
            self.eps_weight,
            self.sigma_0,
            self.N,
            self.alpha,
        )
        bias = None
        if self.mu_bias is not None:
            bias = noise_fn(
                self.mu_bias,
                self.sigma_bias,
                self.eps_bias,
                self.sigma_0,
                self.N,
                self.alpha,
            )
        return F.linear(input, weight, bias)

    def forward_(self, input):
        sig_weight = torch.exp(self.sigma_weight)
        weight = self.mu_weight + sig_weight * self.eps_weight.normal_()
        kl_weight = (
            math.log(self.sigma_0)
            - self.sigma_weight
            + (sig_weight ** 2 + self.mu_weight ** 2) / (2 * self.sigma_0 ** 2)
            - 0.5
        )
        if self.mu_bias is not None:
            sig_bias = torch.exp(self.sigma_bias)
            bias = self.mu_bias + sig_bias * self.eps_bias.normal_()
            kl_bias = (
                math.log(self.sigma_0)
                - self.sigma_bias
                + (sig_bias ** 2 + self.mu_bias ** 2) / (2 * self.sigma_0 ** 2)
                - 0.5
            )
        out = F.linear(input, weight, bias)
        kl = (
            kl_weight.sum() + kl_bias.sum()
            if self.mu_bias is not None
            else kl_weight.sum()
        )
        return out, kl
