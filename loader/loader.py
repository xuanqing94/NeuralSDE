import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

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

def get_cifar_loaders(batch_size, version='cifar10'):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    if version == 'cifar10':
        tr_data = datasets.CIFAR10(root='~/data/cifar10-py', train=True,
            download=True, transform=transform_train)
        te_data = datasets.CIFAR10(root='~/data/cifar10-py', train=False,
            download=True, transform=transform_test)
    elif version == 'cifar10.1':
        # cifar10.1 does not contain training set
        tr_data = datasets.CIFAR10(root='~/data/cifar10-py', train=True,
            download=True, transform=transform_train)
        te_data = np.load('/home/luinx/data/cifar10.1-py/cifar10.1_v6_data.npy') / 255.0
        te_labels = np.load('/home/luinx/data/cifar10.1-py/cifar10.1_v6_labels.npy')
        data = torch.from_numpy(te_data).float()
        data = data.permute(0, 3, 1, 2)
        labels = torch.from_numpy(te_labels).long()
        te_data = TensorDataset(data, labels)
    train_loader = DataLoader(tr_data, batch_size=batch_size,
        shuffle=True, num_workers=2, drop_last=True
    )
    test_loader = DataLoader(te_data, batch_size=batch_size,
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

