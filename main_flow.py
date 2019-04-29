""" Train a flow model based on SDE solver """
import os
import math
import argparse
import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torchvision.transforms import *
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import torch.optim as optim
from models.flow_model import FlowModel 

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument("--data", choices=["cifar10"], type=str, default="cifar10")
parser.add_argument("--dims", type=str, default="8,32,32,8")
parser.add_argument("--strides", type=str, default="2,2,1,-2,-2")
parser.add_argument("--num_blocks", type=int, default=1, help='Number of stacked CNFs.')
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--resume", type=str, default="")
parser.add_argument("--img_dir", type=str, default="./img")
parser.add_argument("--ckpt", type=str, default="./ckpt/flow_model.ckpt")
parser.add_argument("--time_length", type=float, default=1.0)
parser.add_argument("--lr", type=float, default=0.1)
args = parser.parse_args()


def add_noise(x):
    """
    [0, 1] -> [0, 255] -> add noise -> [0, 1]
    """
    noise = x.new().resize_as_(x).uniform_()
    x = x * 255 + noise
    x = x / 256
    return x

def pdf_normal(z):
    logZ = -0.5 * math.log(2 * math.pi)
    return logZ - z.pow(2) / 2

def get_data():
    if args.data == "cifar10":
        tsf_tr = Compose([
            Resize(32),
            RandomHorizontalFlip(),
            ToTensor(),
            add_noise,
        ])
        tsf_te = Compose([
            Resize(32),
            ToTensor(),
            add_noise,
        ])
        train_data = CIFAR10(root="~/data/cifar10-py", train=True, transform=tsf_tr, download=False)
        test_data = CIFAR10(root="~/data/cifar10-py", train=False, transform=tsf_te, download=False)
        data_shape = (3, 32, 32)
    elif args.data == "mnist":
        tsf_tr = Compose([
            Resize(28),
            RandomHorizontalFlip(),
            ToTensor(),
            add_noise,
        ])
        tsf_te = Compose([
            Resize(28),
            ToTensor(),
            add_noise,
        ])
        train_data = MNIST(root="~/data/mnist", train=True, transform=tsf_tr, download=False)
        test_data = MNIST(root="~/data/mnist", train=False, transform=tsf_te, download=False)
        data_shape = (1, 28, 28)
    else:
        raise ValueError("Dataset not supported")

    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False, drop_last=True)
    return train_loader, test_loader, data_shape


def get_model(data_shape):
    hidden_dims = tuple(map(int, args.dims.split(",")))
    strides = tuple(map(int, args.strides.split(",")))
    model = FlowModel((args.batch_size, *data_shape), n_blocks=args.num_blocks,
            intermediate_dims=hidden_dims, cnf_kwargs={"T": args.time_length})
    return model 

def get_opt(params, lr):
    opt = optim.Adam(params, lr=lr, weight_decay=0.0)
    return opt

def loss_f(model, image):
    # initial logp = zeros
    zero = torch.zeros(image.size(0), 1).to(image)
    z, delta_logp = model(image, zero)
    # calculate log density of z
    logpz = pdf_normal(z).view(z.size(0), -1).sum(1, keepdim=True)
    logpx = logpz - delta_logp
    logpx_per_dim = torch.sum(logpx) / image.numel()
    bits_per_dim = -(logpx_per_dim - math.log(256.0)) / math.log(2.0)
    return bits_per_dim

if __name__ == "__main__":
    # get dataset
    train_loader, test_loader, data_shape = get_data()
    # get model
    model = get_model(data_shape)
    model = nn.DataParallel(model).cuda()
    if args.resume != "":
        model.load_state_dict(torch.load(args.resume))
    
    # for visualization
    fixed_z = torch.randn(100, *data_shape).cuda()
    
    # start training
    epochs = [30, 30, 30, 30]
    counter = 0
    for epoch in epochs:
        opt = get_opt(model.parameters(), lr=args.lr)
        args.lr /= 10
        for _ in range(epoch):
            counter += 1
            # training epoch
            for it, (img, _) in enumerate(train_loader):
                opt.zero_grad()
                img = img.cuda()
                # forward prop: img ---> noise
                loss = loss_f(model, img)
                # backward prop
                loss.backward()
                opt.step()
                print(f"Iter {it}, loss: {loss.item()}")
            # testing epoch
            total_loss = 0
            for img, _ in test_loader:
                img = img.cuda()
                with torch.no_grad():
                    loss = loss_f(model, img)
                    total_loss += loss.item()
            total_loss /= len(test_loader)
            print(f"[counter/np.sum(epochs)] Test loss: {total_loss}")
            # visualize
            out_f = os.path.join(args.img_dir, f"fig_{counter}.jpg")
            with torch.no_grad():
                samples = model(fixed_z, reverse=True).view(-1, *data_shape)
            save_image(samples, out_f, nrow=10)
            # save model
            torch.save(model.state_dict(), args.ckpt)
