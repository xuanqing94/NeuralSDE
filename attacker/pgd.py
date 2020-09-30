import math
import torch
import torch.nn.functional as F
from .linf_sgd import Linf_SGD
from torch.optim import Adam, SGD

# performs Linf-constraint PGD attack w/o noise
# @epsilon: radius of Linf-norm ball
def Linf_PGD(x_in, y_true, net, steps, eps, num_avg=1):
    if eps == 0:
        return x_in
    training = net.training
    if training:
        net.eval()
    x_adv = x_in.clone().requires_grad_()
    # NOTE we do multiple forward-backward, so lr should be divided by num_avg
    optimizer = Linf_SGD([x_adv], lr=0.007 / num_avg)
    for _ in range(steps):
        optimizer.zero_grad()
        net.zero_grad()
        for _ in range(num_avg):
            # accumulate gradients for minibatch gd
            out = net(x_adv)
            loss = -F.cross_entropy(out, y_true)
            loss.backward()
        optimizer.step()
        diff = x_adv - x_in
        diff.clamp_(-eps, eps)
        x_adv.detach().copy_((diff + x_in).clamp_(0, 1))
    net.zero_grad()
    # reset to the original state
    if training:
        net.train()
    return x_adv



def L2_PGD(x_in, y_true, net, steps, eps, num_avg=1):
    if eps == 0:
        return x_in

    training = net.training
    if training:
        net.eval()

    x_adv = x_in.clone().requires_grad_()
    #lr = 1.5 * eps / steps / num_avg
    lr = (2 * eps / steps) / num_avg
    optimizer = Adam([x_adv], lr=lr)
    eps = torch.tensor(eps).to(x_in)

    for step in range(steps):
        optimizer.zero_grad()
        net.zero_grad()
        for _ in range(num_avg):
            out = net(x_adv)
            loss = -F.cross_entropy(out, y_true)
            loss.backward()
        optimizer.step()
        diff = x_adv - x_in
        norm = torch.sqrt(torch.sum(diff * diff, (1,2,3)))
        norm = norm.view(norm.size(0), 1, 1, 1)
        norm_out = torch.min(norm, eps)
        diff = diff / norm * norm_out # projection
        x_adv.detach().copy_((diff + x_in).clamp_(0, 1))

    net.zero_grad()
    if training:
        net.train()
    return x_adv

