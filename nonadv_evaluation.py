import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from models.sde_model import SdeClassifier
from dataset.cifar10_c import CIFAR10_C, CIFAR10_C_data


parser = argparse.ArgumentParser(description='')
parser.add_argument('--data', type=str, required=True)
parser.add_argument('--batch_size', type=int, default=200)
parser.add_argument('--sigma', type=float, default=0.0)
parser.add_argument('--test_sigma', type=float, default=0.0)
parser.add_argument('--noise_type', type=str, required=True)
parser.add_argument('--n_ensemble', type=int, required=True)
opt = parser.parse_args()


def loader_from_data(data):
    loader = DataLoader(data, batch_size=opt.batch_size, shuffle=True, num_workers=2)
    return loader

def test_data(n_class, loader, net):
    correct = 0
    total = 0
    softmax = nn.Softmax(dim=1)
    for x, y in loader:
        x, y = x.cuda(), y.cuda()
        batch = y.numel()
        prob = torch.zeros(batch, n_class).cuda()
        # ensemble prediction
        with torch.no_grad():
            for _ in range(opt.n_ensemble):
                p = softmax(net(x))
                prob.add_(p)
            pred = torch.max(prob, dim=1)[1]
        # result
        correct += torch.sum(y.eq(pred)).item()
        total += y.numel()
    return correct / total

if __name__ == "__main__":
    # get dataset
    if opt.data == "cifar10":
        in_nc = 3
        n_class = 10
        datasets = CIFAR10_C("~/data/CIFAR-10-C")
        ckpt_f = f'./ckpt/sde_{opt.data}_{opt.sigma}_{opt.noise_type}.pth'
    else:
        raise ValueError('invalid data')
    
    # prepare model
    net = SdeClassifier(in_nc, opt.test_sigma, mid_state=None, noise_type=opt.noise_type)
    net.load_state_dict(torch.load(ckpt_f))
    net.cuda()
    
    # testing
    for idx in range(datasets.num_datasets()):
        # test data one by one
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
        data = datasets.get_ith_data(idx, transform=transform_test)
        loader = loader_from_data(data)
        # testing for this data
        print(test_data(n_class, loader, net))
