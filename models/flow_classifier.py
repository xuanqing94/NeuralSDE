import torch
import torch.nn as nn

from .diffusion_fn import MultiplicativeNoise, AdditiveNoise, NoNoise
from .jump_fn import PoissonBernoulliJump, NoJump
from .integrated_flow import IntegratedFlow
from .flow_fn import FlowFn, FlowFn_v2
from .flow_net import MultiScaleFlow


def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)


def heavy_downsampling(in_nc, nc_hidden):
    downsampling_layers = [
        nn.Conv2d(in_nc, nc_hidden, 3, 1, 1),
        norm(nc_hidden),
        nn.ReLU(inplace=True),
        nn.Conv2d(nc_hidden, nc_hidden * 2, 3, 1, 1),
        norm(nc_hidden * 2),
        nn.ReLU(inplace=True),
        nn.Conv2d(nc_hidden * 2, nc_hidden * 4, 4, 2, 1),
    ]
    return downsampling_layers, nc_hidden * 4


def light_downsampling(in_nc, nc_hidden):
    downsampling_layers = [nn.Conv2d(in_nc, nc_hidden, 3, 1)]
    return downsampling_layers, nc_hidden


class SdeClassifier(nn.Module):
    def __init__(
        self,
        n_scale,
        nclass,
        nc,
        nc_hidden,
        sigma,
        drop_rate,
        drop_scale,
        grid_size,
        T,
        downsampling_type="heavy",
        noise_type="additive",
        version='v1',
    ):
        super().__init__()
        if downsampling_type == "light":
            layers, nc_hidden = light_downsampling(nc, nc_hidden)
            self.downsampling_layers = nn.Sequential(*layers)
        elif downsampling_type == "heavy":
            layers, nc_hidden = heavy_downsampling(nc, nc_hidden)
            self.downsampling_layers = nn.Sequential(*layers)
        else:
            raise ValueError("Invalid value of downsampling_type")
        if noise_type == "multiplicative":
            diffusion = MultiplicativeNoise(sigma)
            jump = None
        elif noise_type == "additive":
            diffusion = AdditiveNoise(sigma)
            jump = None
        elif noise_type == "dropout":
            diffusion = None
            jump = PoissonBernoulliJump(drop_rate, drop_scale)
        else:
            raise ValueError("Invalid value of noise_type")
        flows = []
        for _ in range(n_scale):
            flow_fn = FlowFn(nc_hidden) if version == 'v1' else FlowFn_v2(nc_hidden)
            flows.append(IntegratedFlow(flow_fn, diffusion, jump, grid_size, T))
        self.multiscale_flows = MultiScaleFlow(flows)
        self.fc_layers = nn.Sequential(
            norm(nc_hidden),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            nn.Linear(nc_hidden, nclass),
        )

    def forward(self, x):
        out = self.downsampling_layers(x)
        out = self.multiscale_flows(out)
        out = self.fc_layers(out)
        return out


if __name__ == "__main__":
    classifier = SdeClassifier(
        n_scale=3, nclass=10, nc=3, nc_hidden=64, sigma=0.1, grid_size=0.1, T=1.0
    )
    x = torch.randn(13, 3, 32, 32)
    out = classifier(x)
    print(out.size())
