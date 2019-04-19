""" 
Modified from rtqichen/ffjord 
BaseNet --> FlowNet --> CNF --> StackedCNFLayers --> FlowModel
"""
import copy
import torch
import torch.nn as nn
import numpy as np
from .utils import ConcatConv2d, sample_rademacher_like, \
        divergence_approx, SequentialFlow, SqueezeLayer, \
        LogitTransform
from sdeint.milstein import sdeint_joint_milstein 
from torchdiffeq import odeint_adjoint as odeint


class BaseNet(nn.Module):
    """
    Helper class to make neural nets for use in continuous normalizing flows
    Edit: remove unnecessary parameters
    """

    def __init__(self, hidden_dims, input_shape, strides):
        super(BaseNet, self).__init__()
        
        assert len(strides) == len(hidden_dims) + 1
        base_layer = ConcatConv2d

        # build layers and add them
        layers = []
        activation_fns = []
        hidden_shape = input_shape

        for dim_out, stride in zip(hidden_dims + (input_shape[0],), strides):
            if stride is None:
                layer_kwargs = {}
            elif stride == 1:
                layer_kwargs = {"ksize": 3, "stride": 1, "padding": 1, "transpose": False}
            elif stride == 2:
                layer_kwargs = {"ksize": 4, "stride": 2, "padding": 1, "transpose": False}
            elif stride == -2:
                layer_kwargs = {"ksize": 4, "stride": 2, "padding": 1, "transpose": True}
            else:
                raise ValueError('Unsupported stride: {}'.format(stride))

            layer = base_layer(hidden_shape[0], dim_out, **layer_kwargs)
            layers.append(layer)
            activation_fns.append(nn.Softplus())

            hidden_shape = list(copy.copy(hidden_shape))
            hidden_shape[0] = dim_out
            if stride == 2:
                hidden_shape[1], hidden_shape[2] = hidden_shape[1] // 2, hidden_shape[2] // 2
            elif stride == -2:
                hidden_shape[1], hidden_shape[2] = hidden_shape[1] * 2, hidden_shape[2] * 2

        self.layers = nn.ModuleList(layers)
        self.activation_fns = nn.ModuleList(activation_fns[:-1])

    def forward(self, t, y):
        dx = y
        for l, layer in enumerate(self.layers):
            dx = layer(t, dx)
            # if not last layer, use nonlinearity
            if l < len(self.layers) - 1:
                dx = self.activation_fns[l](dx)
        return dx

class FlowNet(nn.Module):
    """ Encapsule BaseNet to add Jacobian information """
    def __init__(self, base_net):
        super(FlowNet, self).__init__()
        self.diffeq = base_net
        self.divergence_fn = divergence_approx
        self.sigma_h = .0
        self.sigma_logp = .0

    def reset(self):
        self._e = None

    def forward(self, t, states):
        """ states[0] == hidden feature, states[1] == log p_t """
        y = states[0]

        # convert to tensor
        t = torch.tensor(t).type_as(y)
        batchsize = y.shape[0]

        # Sample and fix the noise.
        if self._e is None:
            self._e = sample_rademacher_like(y)

        with torch.set_grad_enabled(True):
            y.requires_grad_(True)
            t.requires_grad_(True)
            dy = self.diffeq(t, y)
            divergence = self.divergence_fn(dy, y, e=self._e).view(batchsize, 1)
        return dy, -divergence
    
    def diffusion(self, t, states):
        """ adding gaussian noise, here we assume multiplicative noise """
        y, logp = states
        return self.sigma_h * y, self.sigma_logp * logp

    def dif_g(self, t, states):
        """ gradient of diffusion on states """
        return self.sigma_h, self.sigma_logp

class CNF(nn.Module):
    def __init__(self, flow_net, T=1.0, h=0.02):
        super(CNF, self).__init__()
        self.end_time = T
        self.flow_net = flow_net
        self.h = h

    def forward(self, z, logpz=None, reverse=False):
        if logpz is None:
            _logpz = torch.zeros(z.shape[0], 1).to(z)
        else:
            _logpz = logpz
        if reverse:
            t0, t1 = self.end_time, 0.0
        else:
            t0, t1 = 0.0, self.end_time
        # Refresh the odefunc statistics.
        self.flow_net.reset()
        # integration
        # switch methods for debugging purpose
        if True:
            z_t, logpz_t = sdeint_joint_milstein(self.flow_net.forward, self.flow_net.diffusion, self.flow_net.dif_g, t0, t1, self.h, (z, _logpz))
        else:
            state_t = odeint(self.flow_net, (z, _logpz), torch.tensor([t0, t1]).to(z), atol=1.0e-5, rtol=1.0e-5)
            state_t = tuple(s[1] for s in state_t)
            z_t, logpz_t = state_t[:2]

        if logpz is not None:
            return z_t, logpz_t
        else:
            return z_t


class StackedCNFLayers(SequentialFlow):
    def __init__(self, initial_size, idims=(32,), squeeze=True, init_layer=None, n_blocks=1, cnf_kwargs={}):
        strides = tuple([1] + [1 for _ in idims])
        chain = []
        if init_layer is not None:
            chain.append(init_layer)

        def _make_odefunc(size):
            net = BaseNet(idims, size, strides)
            f = FlowNet(net)
            return f

        if squeeze:
            c, h, w = initial_size
            after_squeeze_size = c * 4, h // 2, w // 2
            pre = [CNF(_make_odefunc(initial_size), **cnf_kwargs) for _ in range(n_blocks)]
            post = [CNF(_make_odefunc(after_squeeze_size), **cnf_kwargs) for _ in range(n_blocks)]
            chain += pre + [SqueezeLayer(2)] + post
        else:
            chain += [CNF(_make_odefunc(initial_size), **cnf_kwargs) for _ in range(n_blocks)]

        super(StackedCNFLayers, self).__init__(chain)


class FlowModel(nn.Module):
    """
    Real NVP for image data. Will downsample the input until one of the
    dimensions is less than or equal to 4.

    Args:
        input_size (tuple): 4D tuple of the input size.
        n_scale (int): Number of scales for the representation z.
        n_resblocks (int): Length of the resnet for each coupling layer.
    """

    def __init__(self, input_size, n_blocks=2, intermediate_dims=(32,), squash_input=True, alpha=0.05, cnf_kwargs=None):
        super(FlowModel, self).__init__()
        self.n_scale = self._calc_n_scale(input_size)
        self.n_blocks = n_blocks
        self.intermediate_dims = intermediate_dims
        self.squash_input = squash_input
        self.alpha = alpha
        self.cnf_kwargs = cnf_kwargs if cnf_kwargs else {}

        if not self.n_scale > 0:
            raise ValueError('Could not compute number of scales for input of' 'size (%d,%d,%d,%d)' % input_size)

        self.transforms = self._build_net(input_size)

        self.dims = [o[1:] for o in self.calc_output_size(input_size)]

    def _build_net(self, input_size):
        _, c, h, w = input_size
        transforms = []
        for i in range(self.n_scale):
            transforms.append(
                StackedCNFLayers(
                    initial_size=(c, h, w),
                    idims=self.intermediate_dims,
                    squeeze=(i < self.n_scale - 1),  # don't squeeze last layer
                    init_layer=LogitTransform(self.alpha) if self.squash_input and i == 0 else None,
                    n_blocks=self.n_blocks,
                    cnf_kwargs=self.cnf_kwargs,
                )
            )
            c, h, w = c * 2, h // 2, w // 2
        return nn.ModuleList(transforms)

    def _calc_n_scale(self, input_size):
        _, _, h, w = input_size
        n_scale = 0
        while h >= 4 and w >= 4:
            n_scale += 1
            h = h // 2
            w = w // 2
        return n_scale

    def calc_output_size(self, input_size):
        n, c, h, w = input_size
        output_sizes = []
        for i in range(self.n_scale):
            if i < self.n_scale - 1:
                c *= 2
                h //= 2
                w //= 2
                output_sizes.append((n, c, h, w))
            else:
                output_sizes.append((n, c, h, w))
        return tuple(output_sizes)

    def forward(self, x, logpx=None, reverse=False):
        if reverse:
            return self._generate(x, logpx)
        else:
            return self._logdensity(x, logpx)

    def _logdensity(self, x, logpx=None):
        _logpx = torch.zeros(x.shape[0], 1).to(x) if logpx is None else logpx
        out = []
        for idx in range(len(self.transforms)):
            x, _logpx = self.transforms[idx].forward(x, _logpx)
            if idx < len(self.transforms) - 1:
                d = x.size(1) // 2
                x, factor_out = x[:, :d], x[:, d:]
            else:
                # last layer, no factor out
                factor_out = x
            out.append(factor_out)
        out = [o.view(o.size()[0], -1) for o in out]
        out = torch.cat(out, 1)
        return out if logpx is None else (out, _logpx)

    def _generate(self, z, logpz=None):
        z = z.view(z.shape[0], -1)
        zs = []
        i = 0
        for dims in self.dims:
            s = np.prod(dims)
            zs.append(z[:, i:i + s])
            i += s
        zs = [_z.view(_z.size()[0], *zsize) for _z, zsize in zip(zs, self.dims)]
        _logpz = torch.zeros(zs[0].shape[0], 1).to(zs[0]) if logpz is None else logpz
        z_prev, _logpz = self.transforms[-1](zs[-1], _logpz, reverse=True)
        for idx in range(len(self.transforms) - 2, -1, -1):
            z_prev = torch.cat((z_prev, zs[idx]), dim=1)
            z_prev, _logpz = self.transforms[idx](z_prev, _logpz, reverse=True)
        return z_prev if logpz is None else (z_prev, _logpz)


