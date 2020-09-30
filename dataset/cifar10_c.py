from PIL import Image
import os
import numpy as np
import torch
import torch.utils.data as data


class CIFAR10_C(object):
    def __init__(self, path, levels=[1,2,3,4,5]):
        path = os.path.expanduser(path)
        self.path = path
        self.levels = levels
        files = os.listdir(path)
        datasets = []
        label = np.load(os.path.join(path, "labels.npy"))
        selected = []
        for l in levels:
            selected.append(label[(l-1)*10000:l*10000])
        label = np.concatenate(selected, axis=0)
        for f in files:
            if f == "labels.npy":
                continue
            result = np.load(os.path.join(path, f))
            selected = []
            for l in levels:
                selected.append(result[(l-1)*10000:l*10000])
            selected = np.concatenate(selected, axis=0)
            datasets.append(selected)
        
        self.datasets = datasets
        self.label = label
    
    def num_datasets(self):
        return len(self.datasets)

    def get_ith_data(self, i, transform=None, target_transform=None):
        return CIFAR10_C_data(self.datasets[i], self.label, transform, target_transform)


class CIFAR10_C_data(data.Dataset):
    def __init__(self, dataset, label, transform=None, target_transform=None):
        assert dataset.shape[0] == len(label)
        self.dataset = dataset
        self.label = np.asarray(label, dtype=int)
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
    data = CIFAR10_C("~/data/CIFAR-10-C", [1])
    d = data.get_ith_data(0)
    print(d[0])
