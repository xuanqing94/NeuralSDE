from PIL import Image
import os
import numpy as np
import torch
import torch.utils.data as data


class CIFAR10_C(object):
    def __init__(self, path):
        path = os.path.expanduser(path)
        self.path = path
        files = os.listdir(path)
        datasets = []
        label = np.load(os.path.join(path, "labels.npy"))
        for f in files:
            if f == "labels.npy":
                continue
            result = np.load(os.path.join(path, f))
            datasets.append(result)
        
        self.datasets = datasets
        self.label = label
    
    def num_datasets(self):
        return len(self.datasets)

    def get_ith_data(self, i, transform=None, target_transform=None):
        return CIFAR10_C_data(self.datasets, self.label, i, transform, target_transform)


class CIFAR10_C_data(data.Dataset):
    def __init__(self, datasets, label, i, transform=None, target_transform=None):
        self.dataset = datasets[i]
        self.label = np.asarray(label, dtype=int)
        self.i = i
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img, target = self.dataset[index], self.label[index] 
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target
    
    def __len__(self):
        return len(self.label)


if __name__ == "__main__":
    data = CIFAR10_C("~/data/CIFAR-10-C")
    d = data.get_ith_data(0)
    print(d[0])
