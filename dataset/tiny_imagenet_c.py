from PIL import Image
import os
import numpy as np
import torch
import torch.utils.data as data

class TinyImageNet_C(object):
    def __init__(self, path, wnid_f, levels=[1,2,3,4,5]):
        path = os.path.expanduser(path)
        self.path = path
        self.wnid_f = wnid_f
        self.levels = levels
        datasets, labels = self._load_dir_structure()
        self.datasets = datasets
        self.labels = labels


    def num_datasets(self):
        return len(self.datasets)

    def get_ith_data(self, i, transform=None, target_transform=None):
        return Tiny_ImageNet_C_data(self.datasets[i], self.labels[i], transform, target_transform)

    def _map_class_to_int(self):
        ids = []
        with open(self.wnid_f, "r") as reader:
            for l in reader:
                ids.append(l.strip())
        ids.sort()
        name2int = {ids[i]: i for i in range(len(ids))}
        return name2int

    def _load_dir_structure(self):
        path = self.path
        datasets = []
        labels = []
        name2int = self._map_class_to_int()
        # read types of corruption
        types_corrupt = os.listdir(path)
        # for each type, load all 5 levels
        for t in types_corrupt:
            dir_type = os.path.join(path, t)
            levels = os.listdir(dir_type)
            dataset = []
            label = []
            for l in levels:
                if int(l) not in self.levels:
                    continue
                dir_level = os.path.join(dir_type, l)
                # for each level, list all classes
                all_label = os.listdir(dir_level)
                for c in all_label:
                    dir_label = os.path.join(dir_level, c)
                    label_id = name2int[c]
                    imgs = os.listdir(dir_label)
                    for img in imgs:
                        img_f = os.path.join(dir_label, img)
                        dataset.append(img_f)
                        label.append(label_id)
            datasets.append(dataset)
            labels.append(label)
        return datasets, labels

class Tiny_ImageNet_C_data(data.Dataset):
    def __init__(self, dataset, label, transform=None, target_transform=None):
        self.dataset = dataset
        self.label = np.asarray(label, dtype=int)
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img_f, target = self.dataset[index], self.label[index] 
        with open(img_f, 'rb') as f:
            img = Image.open(f)
            img_data = img.convert('RGB') 
        
        if self.transform is not None:
            img = self.transform(img)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target
    
    def __len__(self):
        return len(self.label)


if __name__ == "__main__":
    data = TinyImageNet_C("~/data/Tiny-ImageNet-C", "./wnids.txt")
    print(data.num_datasets())
    data0 = data.get_ith_data(0)
    img, target = data0[0]
    print(img, target)
