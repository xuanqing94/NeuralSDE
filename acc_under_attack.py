#!/usr/bin/env python

"""
PGD attack tests, copied & modified from xuanqing94/BayesianDefense
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
import torchvision
import torchvision.transforms as transforms
# models
from models.sde_model import SdeClassifier
from models.ode_model import OdeClassifier
# adversarial algorithm
from attacker.pgd import Linf_PGD, L2_PGD

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

# arguments
parser = argparse.ArgumentParser(description='Accurary under attack')
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--data', type=str, required=True)
parser.add_argument('--n_ensemble', type=int, required=True)
parser.add_argument('--steps', type=int, required=True)
parser.add_argument('--max_norm', type=str, required=True)
parser.add_argument('--sigma', type=float, default=0.0)
parser.add_argument('--test_sigma', type=float, default=0.0)
parser.add_argument('--batch_size', type=int, default=200)

opt = parser.parse_args()

opt.max_norm = [float(s) for s in opt.max_norm.split(',')]

# attack
attack_f = L2_PGD

# dataset
print('==> Preparing data..')
if opt.data == 'cifar10':
    nclass = 10
    img_width = 32
    in_nc = 3
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    testset = torchvision.datasets.CIFAR10(root='~/data/cifar10-py', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size, shuffle=True, num_workers=2)
elif opt.data == "mnist":
    nclass = 10
    img_width = 28
    in_nc = 1
    transfrom_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    testset = torchvision.datasets.MNIST(root='/home/luinx/data/mnist', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size, shuffle=True, num_workers=2)
else:
    raise ValueError(f'invlid dataset: {opt.data}')

# load model
if opt.model == 'ode':
    net = OdeClassifier(in_nc)
    f = f'./ckpt/ode_{opt.data}.pth'
elif opt.model == 'sde':
    net = SdeClassifier(in_nc, opt.test_sigma, mid_state=None)
    f = f'./ckpt/sde_{opt.data}_{opt.sigma}.pth'
else:
    raise ValueError('invalid opt.model')

print(f"Loading model from file {f}")
net.load_state_dict(torch.load(f))
net.cuda()
net.eval() # must set to evaluation mode
loss_f = nn.CrossEntropyLoss()
softmax = nn.Softmax(dim=1)

def ensemble_inference(x_in):
    batch = x_in.size(0)
    prob = torch.FloatTensor(batch, nclass).zero_().cuda()
    with torch.no_grad():
        for _ in range(opt.n_ensemble):
            p = softmax(net(x_in))
            prob.add_(p)
        answer = torch.max(prob, dim=1)[1]
    return answer

def linf_distance(x_adv, x):
    diff = (x_adv - x).view(x.size(0), -1)
    out = torch.mean(torch.max(torch.abs(diff), 1)[0]).item()
    return out

def l2_distance(x_adv, x):
    diff = (x_adv - x).view(x.size(0), -1)
    out = torch.mean(torch.norm(diff, dim=1)).item()
    return out


# Iterate over test set
print('#norm, accuracy')
for eps in opt.max_norm:
    correct = 0
    total = 0
    max_iter = 40
    distortion = 0
    batch = 0
    for it, (x, y) in enumerate(testloader):
        x, y = x.cuda(), y.cuda()
        x_adv = attack_f(x, y, net, opt.steps, eps)
        pred = ensemble_inference(x_adv)
        correct += torch.sum(pred.eq(y)).item()
        total += y.numel()
        distortion += l2_distance(x_adv, x)
        batch += 1
        if it >= max_iter:
            break
    print(f'{distortion/batch}, {correct/total}')

