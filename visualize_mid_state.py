#!/usr/bin/env python

import torch
import torchvision
import torchvision.transforms as transforms
from models.sde_model import SdeClassifier
from attacker.pgd import L2_PGD

data = 'cifar10'
in_nc = 3
sigma = 0.0
test_sigma = 0.0

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
    net = SdeClassifier(in_nc, test_sigma, None, 0.02)
    f = f'./ckpt/sde_{data}_{sigma}.pth'
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
        testset = torchvision.datasets.CIFAR10(root='~/data/cifar10-py', train=False, download=True, transform=transform_test)
    
    for x, y in testset:
        x, y = x.unsqueeze(0).cuda(), torch.LongTensor([y]).cuda()
        # calculate the mid states for original input
        mid_state_original = get_middle_states(net, x, 100)
        #net.set_mid_state(mid_state_original)
        #with torch.no_grad():
        #    out = net(x)
        # get adversarial example from (x, y)
        x_adv = L2_PGD(x, y, net, 1000, 0.1)
        mid_state_adv = get_middle_states(net, x_adv, 100)
        #net.set_mid_state(mid_state_adv)
        #with torch.no_grad():
        #    out2 = net(x_adv)
        snr = get_snr(mid_state_original, mid_state_adv)
        print(snr)
        break
