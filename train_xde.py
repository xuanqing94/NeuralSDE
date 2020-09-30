#!/usr/bin/env python

import os
import math
import argparse
import torch
import torch.nn as nn

from models.flow_classifier import SdeClassifier
from models.bayesian_flow_classifier import BayesianClassifier
from loader.loader import get_mnist_loaders, get_cifar_loaders, get_stl_loaders


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training a ODE/SDE classifier")
    parser.add_argument(
        "--n_scale", type=int, default=3, help="Number of scales of resolution."
    )
    parser.add_argument(
        "--data",
        type=str,
        choices=["cifar10", "cifar-10.1", "stl10", "mnist", "tiny-imagenet"],
        default="cifar10",
    )
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size of training.")
    parser.add_argument("--sigma", type=float, default=0.0)
    parser.add_argument("--drop_rate", type=float, default=2.0, help="Rate of dropout.")
    parser.add_argument("--drop_scale", type=float, default=1.0, help="Scale of dropout.")
    parser.add_argument("--epochs", type=str, default="60,30,20")
    parser.add_argument("--grid_size", type=float, default=0.1)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--max_lr", type=float, default=0.1)
    parser.add_argument("--save_to", type=str, default="./checkpoints")
    parser.add_argument("--noise_type", type=str, default="additive")
    #parser.add_argument("--print_model_size", action="store_true")
    parser.add_argument("--go_bayesian", action="store_true", help="If set, the model will be Bayesian.")
    parser.add_argument("--bayesian_alpha", type=float, default=None, help="Weight for KL term in Bayesian training.")
    parser.add_argument("--prior_std", type=float, default=None, help="Stddev of prior distribution.")
    parser.add_argument("--version", type=str, default="v1", help="Version of FlowFn.")
    args = parser.parse_args()
    print(args)
    # Check argument
    if args.go_bayesian:
        assert args.sigma == 0, "For Bayesian training, sigma must be zero"
        assert args.prior_std > 0, "For Bayesian training, prior_std must be valid"
        # We change the noise name for better checkpointing
        args.noise_type = "bayesian"
        args.sigma = args.prior_std
    if args.data in ["cifar10", "cifar-10.1"]:
        train_loader, test_loader = get_cifar_loaders(args.batch_size, version=args.data)
        if args.go_bayesian:
            rand_args = {
                'sigma_0': args.prior_std,
                'N': args.batch_size * len(train_loader), # This is not accurate but works
                'init_s': math.log(args.prior_std),
                'alpha': args.bayesian_alpha,
            }
            model = BayesianClassifier(
                n_scale=args.n_scale,
                nclass=10,
                nc=3,
                nc_hidden=64,
                grid_size=args.grid_size,
                T=args.T,
                downsampling_type="heavy",
                version=args.version,
                **rand_args,
            ).cuda()
        else:
            model = SdeClassifier(
                n_scale=args.n_scale,
                nclass=10,
                nc=3,
                nc_hidden=64,
                sigma=args.sigma,
                drop_rate=args.drop_rate,
                drop_scale=args.drop_scale,
                grid_size=args.grid_size,
                T=args.T,
                downsampling_type="heavy",
                noise_type=args.noise_type,
                version=args.version,
            ).cuda()
    else:
        raise ValueError("Invalid argument of args.data")

    loss_f = nn.CrossEntropyLoss()

    epochs = [int(k) for k in args.epochs.split(',')]
    epoch_counter = 0
    for epoch in epochs:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.max_lr,
                momentum=0.9)
        args.max_lr /= 10
        for k in range(epoch):
            epoch_counter += 1
            train_acc = train_one_epoch(train_loader, model, optimizer, loss_f)
            test_acc = test_one_epoch(test_loader, model, optimizer, loss_f)
            print(f"[Epoch={epoch_counter}] Train: {train_acc*100:.3f}, "
                    f"Test: {test_acc*100:.3f}")
            # save model
            if args.noise_type in ["additive", "multiplicative", "bayesian"]:
                torch.save(model.state_dict(), f"{args.save_to}/sde_{args.data}_{args.sigma}_{args.noise_type}_{args.version}.pth")
            elif args.noise_type in ["dropout"]:
                torch.save(model.state_dict(), f"{args.save_to}/sde_{args.data}_{args.drop_rate}_{args.drop_scale}_{args.noise_type}_{args.version}.pth")

