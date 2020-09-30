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
from models.flow_classifier import SdeClassifier
from models.bayesian_flow_classifier import BayesianClassifier

# adversarial algorithm
from attacker.pgd import Linf_PGD, L2_PGD
from loader.loader import get_mnist_loaders, get_cifar_loaders, get_stl_loaders
import torch.backends.cudnn as cudnn

cudnn.benchmark = True


def linf_distance(x_adv, x):
    diff = (x_adv - x).view(x.size(0), -1)
    out = torch.mean(torch.max(torch.abs(diff), 1)[0]).item()
    return out


def l2_distance(x_adv, x):
    diff = (x_adv - x).view(x.size(0), -1)
    out = torch.mean(torch.norm(diff, dim=1)).item()
    return out


def ensemble_inference(model, x_in, n_ensemble, nclass):
    """Get prediction result by averaging multiple forward"""
    batch = x_in.size(0)
    prob = torch.zeros(batch, nclass).to(x_in)
    softmax = nn.Softmax(dim=1)
    with torch.no_grad():
        for _ in range(n_ensemble):
            p = softmax(model(x_in))
            prob.add_(p)
        answer = torch.max(prob, dim=1)[1]
    return answer


def accuracy_under_attack(test_loader, model, max_norm, num_avg, n_ensemble, nclass):
    correct = 0
    total = 0
    # attack
    attack_f = L2_PGD
    for it, (x, y) in enumerate(test_loader):
        x, y = x.cuda(), y.cuda()
        x_adv = attack_f(x, y, model, args.steps, max_norm, num_avg)
        prediction = ensemble_inference(model, x_adv, n_ensemble, nclass)
        correct += torch.sum(prediction.eq(y)).item()
        total += y.numel()
    return correct / total


def run_attack(test_loader, model, args, nclass):
    print("#Max Linf, Accuracy")
    for max_norm in args.max_norm:
        acc = accuracy_under_attack(test_loader, model, max_norm, args.num_avg, args.n_ensemble, nclass)
        print(max_norm, acc)


if __name__ == "__main__":
    # arguments
    parser = argparse.ArgumentParser(
        description="Evaluating the accurary under adversarial attack"
    )
    parser.add_argument(
        "--n_scale", type=int, default=3, help="NUmber of scales of resolution."
    )
    parser.add_argument(
        "--data",
        type=str,
        choices=["cifar10", "cifar10.1", "stl10", "mnist", "tiny-imagenet"],
        default="cifar10",
    )
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="./checkpoints",
        help="Directory storing checkpoints.",
    )
    parser.add_argument("--sigma", type=float, default=0.0)
    parser.add_argument(
        "--n_ensemble", type=int, default=1, help="Number of forward ensemble."
    )
    parser.add_argument("--drop_rate", type=float, default=0.0, help="Rate of dropout.")
    parser.add_argument("--drop_scale", type=float, default=0.0, help="Scale of dropout.")
    parser.add_argument("--steps", type=int, default=20, help="Number of PGD steps.")
    parser.add_argument(
        "--max_norm", type=str, default="0.0", help="Max Linf perturbation."
    )
    parser.add_argument("--test_sigma", type=float, default=None)
    parser.add_argument(
        "--num_avg", type=int, default=1, help="Number of averaging gradients."
    )
    parser.add_argument("--noise_type", type=str, default="additive")
    parser.add_argument(
        "--go_bayesian", action="store_true", help="If set, the model will be Bayesian."
    )
    parser.add_argument(
        "--prior_std", type=float, default=None, help="Stddev of prior distribution."
    )
    parser.add_argument(
        "--grid_size", type=float, default=0.1, help="Grid size when using Euler method."
    )
    parser.add_argument(
        "--T", type=float, default=1.0, help="Final time."
    )
    parser.add_argument("--version", type=str, default="v1")
    args = parser.parse_args()
    print(args)

    if args.go_bayesian:
        assert args.sigma == 0, "For Bayesian training, sigma must be zero"
        assert args.prior_std > 0, "For Bayesian training, prior_std must be valid"
        # We change then noise name for better checkpointing
        args.noise_type = "bayesian"
        args.sigma = args.prior_std
    # Test sigma and sigma can be different: we can train a model in one sigma
    # and test the model in a different sigma.
    if args.test_sigma is None:
        args.test_sigma = args.sigma
    args.max_norm = [float(s) for s in args.max_norm.split(",")]

    # dataset
    if args.data in ["cifar10", "cifar10.1"]:
        train_loader, test_loader = get_cifar_loaders(args.batch_size, version=args.data)
        nclass = 10
        nc = 3
        # after loading data, need to reset args.data to cifar10
        args.data = "cifar10"
    elif args.data == "stl10":
        train_loader, test_loader = get_stl_loaders(args.batch_size)
        nclass = 10
        nc = 3
    else:
        raise ValueError("Invalid argument of args.data")

    if args.go_bayesian:
        rand_args = {
            "sigma_0": args.prior_std,
            "N": args.batch_size * len(train_loader),  # This is not accurate but works
            "init_s": math.log(args.prior_std),
            "alpha": args.bayesian_alpha,
        }
        model = BayesianClassifier(
            n_scale=args.n_scale,
            nclass=nclass,
            nc=nc,
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
            nclass=nclass,
            nc=nc,
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

    # load model
    if args.noise_type in ["additive", "multiplicative", "bayesian"]:
        ckpt_f = f"checkpoints/sde_{args.data}_{args.sigma}_{args.noise_type}_{args.version}.pth"
        #ckpt_f = f"checkpoints/sde_{args.data}_{args.sigma}_{args.noise_type}.pth"
    elif args.noise_type in ["dropout"]:
        ckpt_f = f"checkpoints/sde_{args.data}_{args.drop_rate}_{args.drop_scale}_{args.noise_type}_{args.version}.pth"

    print(f"Loading model from file {ckpt_f}")
    model.load_state_dict(torch.load(ckpt_f))
    model.eval()  # must set to evaluation mode

    run_attack(test_loader, model, args, nclass)
