# Jump process

import torch
import torch.nn as nn
import pdb

class PoissonBernoulliJump(nn.Module):
    """Implements Poisson - Bernoulli jump process.
    The rate is determined by `rate` parameter.
    """
    def __init__(self, rate, scale):
        super().__init__()
        self.register_buffer("rate", torch.tensor(rate, dtype=torch.float))
        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float))
        self.p = None
        self.dN = None
        self.Z = None
        self.half = None

    def jump(self, dt, t, x):
        # 1. Sample a Bernoulli distribution
        if self.p is None or self.p.size() != x.size():
            self.p = torch.zeros(x.size()).fill_(self.rate * dt)
            self.dN = x.data.clone()
            self.Z = x.data.clone()
            self.half = x.data.clone().fill_(0.5)

        #pdb.set_trace()
        torch.bernoulli(self.p, out=self.dN)
        # 2. Sample jump direction
        torch.bernoulli(self.half, out=self.Z)
        self.Z = 2.0 * self.Z - 1.0
        # 3. Return result
        return self.scale.data * x * self.Z.data * self.dN.data

class NoJump(nn.Module):
    def __init__(self, rate, scale):
        super().__init__()

    def jump(self, dt, t, x):
        return 0.0
