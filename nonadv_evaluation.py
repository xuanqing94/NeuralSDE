#!/usr/bibn/env python

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD

import torchvision
import torchvision.transforms as transforms

# models
from models.flow_classifier import SdeClassifier
from dataset.cifar10_c import CIFAR10_C
from dataset.tiny_imagenet_c import TinyImageNet_C

import torch.backends.cudnn as cudnn
cudnn.benchmark = True


def loader_from_data(data):
    loader = DataLoader(data, batch_size=opt.batch_size, shuffle=True, num_workers=2)
    return loader

def ensemble_inference(model, x_in, n_ensemble, nclass):
    """Get prediction nresult by averaging multiple forward"""
    batch = x_in.size(0)
    prob = torch.zeros(batch, nclass).to(x_in)
    softmax = nn.Softmax(dim=1)
    with torch.no_grad():
        for _ in range(n_ensemble):
            p = softmax(model(x_in))
            prob.add_(p)
        answer = torch.max(prob, dim=1)[1]
    return answer


def test_data(test_loader, model, n_ensemble, nclass, max_batch=30):
    correct = 0
    total = 0
    for i, (x, y) in enumerate(loader):
        if i == max_batch:
            break
        x, y = x.cuda(), y.cuda()
        prediction = ensemble_inference(model, x, n_ensemble, nclass)
        correct += torch.sum(prediction.eq(y)).item()
        total += y.numel()
    return correct / total

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        "--n_scale", type=int, default=3, help="Number of scales of resolution."
    )
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--sigma', type=float, default=0.0)
    parser.add_argument('--test_sigma', type=float, default=0.0)
    parser.add_argument('--noise_type', type=str, required=True)
    parser.add_argument('--levels', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--n_ensemble', type=int, required=True)
    opt = parser.parse_args()

    opt.levels = [int(l) for l in opt.levels.split(',')]
    print("Running under levels = ", opt.levels)

    # get dataset
    if opt.data == "cifar10":
        in_nc = 3
        n_class = 10
        datasets = CIFAR10_C("~/data/CIFAR-10-C", opt.levels)
        ckpt_f = opt.ckpt #f'./ckpt/sde_{opt.data}_{opt.sigma}_{opt.noise_type}.pth'
        Model = SdeClassifier
    elif opt.data == "tiny-imagenet":
        in_nc = 3
        n_class =200
        datasets = TinyImageNet_C("~/data/Tiny-ImageNet-C", "./dataset/wnids.txt", opt.levels)
        ckpt_f = opt.ckpt
        Model = SdeClassifier_big
    else:
        raise ValueError('invalid data')

    # prepare model
    net = Model(in_nc, opt.test_sigma, mid_state=None, noise_type=opt.noise_type)
    net.load_state_dict(torch.load(ckpt_f))
    net.cuda()
    net = nn.DataParallel(net)
    mean_acc = 0
    # testing
    for idx in range(datasets.num_datasets()):
        # test data one by one
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
        data = datasets.get_ith_data(idx, transform=transform_test)
        loader = loader_from_data(data)
        # testing for this data
        acc = test_data(n_class, loader, net)
        mean_acc += acc
    print(f"===> Mean_acc = {mean_acc / datasets.num_datasets()}") 
