#!/usr/bin/env python

import argparse
import torch
import torchvision
import torchvision.transforms as transforms
from models.sde_model import SdeClassifier
from attacker.pgd import L2_PGD
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, choices=['cifar10', 'mnist'], default='mnist')
parser.add_argument('--sigma', type=float, default=0.0)
parser.add_argument('--test_sigma', type=float, default=0.0)
parser.add_argument('--step_size', type=float, default=0.2)
parser.add_argument('--n_ensemble', type=int, default=2000)
parser.add_argument('--noise_type', type=str, required=True)
args = parser.parse_args()

data = args.data
in_nc = 3 if args.data == 'cifar10' else 1
sigma = args.sigma
test_sigma = args.test_sigma

def get_middle_states(net, x, n_try):
    avg_result = []
    for _ in range(n_try):
        mid_state_original = []
        net.set_mid_state(mid_state_original)
        with torch.no_grad():
            out = net(x)
        for layer, state in enumerate(mid_state_original):
            if len(avg_result) < len(mid_state_original):
                avg_result.append(state)
            else:
                avg_result[layer] += state
    avg_result = [k / n_try for k in avg_result]
    return avg_result

def get_norm(mid_states):
    out = []
    for state in mid_states:
        out.append(torch.norm(state).item())
    return out

def get_snr(states1, states2):
    out = []
    for state1, state2 in zip(states1, states2):
        diff = torch.norm(state1 - state2).item()
        norm = torch.norm(state1).item()
        out.append(diff / norm)
    return out

if __name__ == "__main__":
    # load model
    net = SdeClassifier(in_nc, test_sigma, None, args.step_size, args.noise_type)
    f = f'./ckpt/sde_{data}_{sigma}_{args.noise_type}.pth'
    net.load_state_dict(torch.load(f))
    net.cuda()
    net.eval()
    # get one data example
    if data == 'cifar10':
        nclass = 10
        img_width = 32
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
        testset = torchvision.datasets.CIFAR10(root='~/data/cifar10-py', train=False, download=False, transform=transform_test)
    
    for x, y in testset:
        x, y = x.unsqueeze(0).cuda(), torch.LongTensor([y]).cuda()
        # calculate the mid states on original input
        mid_state_original = get_middle_states(net, x, args.n_ensemble)
        # get adversarial example from (x, y)
        x_adv = L2_PGD(x, y, net, 40, 0.1)
        # calculate the mid states on adversarial input
        mid_state_adv = get_middle_states(net, x_adv, args.n_ensemble)
        # caluclate relative change of middle states
        snr = get_snr(mid_state_original, mid_state_adv)
        print(snr)
        # we only calculate one input
        # TODO maybe average over multiple inputs?
        break
