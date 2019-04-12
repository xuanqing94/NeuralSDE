#!/usr/bin/env python

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from models.ode_model import OdeClassifier

parser = argparse.ArgumentParser()
parser.add_argument('--network', type=str, choices=['resnet', 'odenet'], default='odenet')
parser.add_argument('--data', type=str, choices=['cifar10', 'mnist'], default='mnist')
parser.add_argument('--tol', type=float, default=1e-3)
parser.add_argument('--adjoint', type=eval, default=False, choices=[True, False])
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--save', type=str, default='./experiment1')
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint


def get_mnist_loaders():
    transform_train = transforms.Compose([
        transforms.RandomCrop(28, padding=4),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_loader = DataLoader(
        datasets.MNIST(root='/home/luinx/data/mnist', train=True,
            download=True, transform=transform_train), batch_size=128,
        shuffle=True, num_workers=2, drop_last=True
    )
    test_loader = DataLoader(
        datasets.MNIST(root='/home/luinx/data/mnist', train=False,
            download=True, transform=transform_test), batch_size=128,
        shuffle=False, num_workers=2, drop_last=True
    )
    return train_loader, test_loader

def get_cifar_loaders():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_loader = DataLoader(
        datasets.CIFAR10(root='/home/luinx/data/cifar10-py', train=True,
            download=True, transform=transform_train), batch_size=128,
        shuffle=True, num_workers=2, drop_last=True
    )
    test_loader = DataLoader(
        datasets.CIFAR10(root='/home/luinx/data/cifar10-py', train=False,
            download=True, transform=transform_test), batch_size=128,
        shuffle=False, num_workers=2, drop_last=True
    )
    return train_loader, test_loader

def train_one_epoch(loader, model, optimizer, loss_f):
    model.train()
    total = 0
    correct = 0
    for x, y in loader:
        # forward / backward
        x, y = x.cuda(), y.cuda()
        optimizer.zero_grad()
        output = model(x)
        loss = loss_f(output, y)
        loss.backward()
        optimizer.step()
        # gather stats
        correct += y.eq(torch.max(output, dim=1)[1]).sum().item()
        total += y.numel()
    return correct / total

def test_one_epoch(loader, model, optimizer, loss_f):
    model.eval()
    total = 0
    correct = 0
    for x, y in loader:
        # forward / backward
        x, y = x.cuda(), y.cuda()
        with torch.no_grad():
            output = model(x)
        # gather stats
        correct += y.eq(torch.max(output, dim=1)[1]).sum().item()
        total += y.numel()
    return correct / total

if __name__ == '__main__':
    if args.data == "cifar10":
        model = OdeClassifier(in_nc=3).cuda()
        train_loader, test_loader = get_cifar_loaders()
    elif args.data == "mnist":
        model = OdeClassifier(in_nc=1).cuda()
        train_loader, test_loader = get_mnist_loaders()

    loss = nn.CrossEntropyLoss()
    epochs = [30, 20, 10]
    epoch_counter = 0
    for epoch in epochs:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                momentum=0.9)
        args.lr /= 10
        epoch_counter += 1
        for k in range(epoch):
            train_acc = train_one_epoch(train_loader, model, optimizer, loss)
            test_acc = test_one_epoch(test_loader, model, optimizer, loss)
            print(f"[Epoch={epoch_counter}] Train: {train_acc:.3f}, "
                    f"Test: {test_acc:.3f}")
    torch.save(model.state_dict(), f"./ckpt/ode_{args.data}.pth")
