#!/usr/bin/env python

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from models.resnet import ResNet_S, ResNet_L

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, choices=['cifar10', 'stl10', 'mnist', 'tiny-imagenet'], default='mnist')
parser.add_argument('--epochs', type=str, default="20,10,10,5")
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--model_size', action='store_true')
args = parser.parse_args()

from sdeint.euler import sdeint_euler

def get_mnist_loaders():
    transform_train = transforms.Compose([
        transforms.RandomCrop(28, padding=4),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_loader = DataLoader(
        datasets.MNIST(root='~/data/mnist', train=True,
            download=True, transform=transform_train), batch_size=128,
        shuffle=True, num_workers=2, drop_last=True
    )
    test_loader = DataLoader(
        datasets.MNIST(root='~/data/mnist', train=False,
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
        datasets.CIFAR10(root='~/data/cifar10-py', train=True,
            download=True, transform=transform_train), batch_size=128,
        shuffle=True, num_workers=2, drop_last=True
    )
    test_loader = DataLoader(
        datasets.CIFAR10(root='~/data/cifar10-py', train=False,
            download=True, transform=transform_test), batch_size=128,
        shuffle=False, num_workers=2, drop_last=True
    )
    return train_loader, test_loader

def get_stl_loaders():
    transform_train = transforms.Compose([
        transforms.RandomCrop(96, padding=8),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_loader = DataLoader(
        datasets.STL10(root='~/data/stl10', split='train',
            download=True, transform=transform_train), batch_size=64,
        shuffle=True, num_workers=2, drop_last=True
    )
    test_loader = DataLoader(
        datasets.STL10(root='~/data/stl10', split='test',
            download=True, transform=transform_test), batch_size=64,
        shuffle=False, num_workers=2, drop_last=True
    )
    return train_loader, test_loader


def get_tiny_imagenet_loaders():
    transform_train = transforms.Compose([
        transforms.RandomCrop(64, padding=6),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_loader = DataLoader(
        datasets.ImageFolder(root='/home/luinx/data/Tiny-ImageNet/train', 
            transform=transform_train),
        batch_size=100, shuffle=True, num_workers=2, drop_last=True
    )
    test_loader = DataLoader(
        datasets.ImageFolder(root='/home/luinx/data/Tiny-ImageNet/val',
            transform=transform_test),
        batch_size=100, shuffle=False, num_workers=2, drop_last=True
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
        model = ResNet_S().cuda()
        train_loader, test_loader = get_cifar_loaders()
    elif args.data == "stl10":
        model = ResNet_L().cuda()
        train_loader, test_loader = get_stl_loaders()
    elif args.data == "tiny-imagenet":
        model = ResNet_L().cuda()
        train_loader, test_loader = get_tiny_imagenet_loaders()
    else:
        raise ValueError
    
    model = nn.DataParallel(model)
    if args.model_size:
        n_params = 0
        for t in model.parameters():
            n_params += t.numel()
        print(f"===> Total number of params: {n_params}")
        exit(-1)
    loss = nn.CrossEntropyLoss()
    epochs = [int(k) for k in args.epochs.split(',')]
    epoch_counter = 0
    for epoch in epochs:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                momentum=0.9)
        args.lr /= 10
        for k in range(epoch):
            epoch_counter += 1
            train_acc = train_one_epoch(train_loader, model, optimizer, loss)
            test_acc = test_one_epoch(test_loader, model, optimizer, loss)
            print(f"[Epoch={epoch_counter}] Train: {train_acc:.3f}, "
                    f"Test: {test_acc:.3f}")
            # save model
            torch.save(model.module.state_dict(), f"./ckpt/resnet_{args.data}.pth")
