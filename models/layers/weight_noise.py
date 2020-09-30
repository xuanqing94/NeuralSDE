import torch
import torch.nn.functional as F
from torch.autograd import Function


class NoiseFn(Function):
    @staticmethod
    def forward(ctx, mu, sigma, eps, sigma_0, N, alpha):
        """Forward function of random layer.
        Args:
            ctx: context.
            mu: Mean of weights.
            sigma: log std of weights (the actual std is exp(sigma)).
            eps: standard normal, will be resampled at every iteration.
            sigma_0: prior of standard deviation.
            N: number of training samples.
            alpha: weighting the regularization term.
        """
        eps = torch.randn_like(eps)
        ctx.save_for_backward(mu, sigma, eps)
        ctx.sigma_0 = sigma_0
        ctx.N = N
        ctx.alpha = alpha
        return mu + torch.exp(sigma) * eps

    @staticmethod
    def backward(ctx, grad_output):
        mu, sigma, eps = ctx.saved_tensors
        sigma_0, N, alpha = ctx.sigma_0, ctx.N, ctx.alpha
        grad_mu = grad_sigma = grad_eps = grad_sigma_0 = grad_N = grad_alpha = None
        tmp = torch.exp(sigma)
        if ctx.needs_input_grad[0]:
            grad_mu = grad_output + alpha * mu / (sigma_0 * sigma_0 * N)
        if ctx.needs_input_grad[1]:
            grad_sigma = (
                grad_output * tmp * eps
                - alpha / N
                + alpha * tmp * tmp / (sigma_0 * sigma_0 * N)
            )
        return grad_mu, grad_sigma, grad_eps, grad_sigma_0, grad_N, grad_alpha


class IdFn(Function):
    @staticmethod
    def forward(ctx, mu, sigma, eps, sigma_0, N, alpha):
        return mu

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None, None


noise_fn = NoiseFn.apply
