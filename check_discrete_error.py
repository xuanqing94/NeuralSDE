#!/usr/bin/env python

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

# models
from models.sde_model import SdeClassifier, SdeClassifier_big

import torch.backends.cudnn as cudnn

cudnn.benchmark = True

parser = argparse.ArgumentParser(
    description="Evaluate the error caused by discretization"
)
parser.add_argument("--data", type=str, default="cifar10")
parser.add_argument("--n_ensemble", type=int, default=1000)
parser.add_argument("--grid_size", type=str, default="0.1")
parser.add_argument("--T", type=float, default=1.0)
parser.add_argument("--noise_type", type=str, default="additive")
parser.add_argument("--test_sigma", type=float, default=10)
opt = parser.parse_args()

opt.grid_size = [float(gs) for gs in opt.grid_size.split(",")]

if opt.data == "cifar10":
    nclass = 10
    img_width = 32
    in_nc = 3
    transform_test = transforms.Compose([transforms.ToTensor()])
    testset = torchvision.datasets.CIFAR10(
        root="~/data/cifar10-py", train=False, download=True, transform=transform_test
    )
    net = SdeClassifier(
        in_nc, noise_type=opt.noise_type
    )

elif opt.data == "mnist":
    nclass = 10
    img_width = 28
    in_nc = 1
    transform_test = transforms.Compose([transforms.ToTensor()])
    testset = torchvision.datasets.MNIST(
        root="/home/luinx/data/mnist",
        train=False,
        download=True,
        transform=transform_test,
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=opt.batch_size, shuffle=False, num_workers=2
    )
    net = SdeClassifier(
        in_nc, opt.test_sigma, mid_state=None, noise_type=opt.noise_type
    )

elif opt.data == "stl10":
    nclass = 10
    img_width = 96
    in_nc = 3
    transform_test = transforms.Compose([transforms.ToTensor()])
    testset = torchvision.datasets.STL10(
        root="~/data/stl10", split="test", download=False, transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=opt.batch_size, shuffle=False, num_workers=2
    )
    net = SdeClassifier_big(
        in_nc, opt.test_sigma, mid_state=None, noise_type=opt.noise_type, n_class=nclass
    )

elif opt.data == "tiny-imagenet":
    nclass = 200
    img_width = 64
    in_nc = 3
    transform_test = transforms.Compose([transforms.ToTensor()])
    testset = torchvision.datasets.ImageFolder(
        root="/home/luinx/data/Tiny-ImageNet/val", transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=opt.batch_size, shuffle=False, num_workers=2
    )
    net = SdeClassifier_big(
        in_nc, opt.test_sigma, mid_state=None, noise_type=opt.noise_type, n_class=nclass
    )

else:
    raise ValueError(f"invlid dataset: {opt.data}")


# load model
f = f"./ckpt_error_analysis/sde_{opt.data}_{opt.test_sigma}_{opt.noise_type}.pth"
print(f"Loading model from file {f}")
net.load_state_dict(torch.load(f))
net.cuda()
net.eval()  # must set to evaluation mode


def average_embedding(x_in, grid):
    # Set 'logits' to False and forward() will return embedding
    kwargs = {"logits": True, "sigma": opt.test_sigma, "T": 1, "grid": grid}
    embedding = 0
    with torch.no_grad():
        for _ in range(opt.n_ensemble):
            with torch.no_grad():
                embedding += net(x_in, kwargs)
    return embedding / opt.n_ensemble

def error(x_est, x_truth):
    return torch.norm(x_est - x_truth) / torch.norm(x_truth)

# Prepare a mini-batch of size 1.
x, y = testset[1]
x.unsqueeze_(0)
x = x.cuda()
# Suppose using grid = 0.0001 is accurate enough
#mean_emb_truth = average_embedding(x, 0.0001)
for grid in opt.grid_size:
    mean_emb = average_embedding(x, grid)
    print(mean_emb)
    top1, idx = torch.max(mean_emb, dim=1)
    #assert(idx.item() == y)
    #err = error(mean_emb, mean_emb_truth)
