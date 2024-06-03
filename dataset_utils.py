
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F
from torchvision import transforms, utils, datasets
from PIL import Image

class resized_dataset(Dataset):
    def __init__(self, dataset, transform=None, start=None, end=None, resize=None):
        self.data=[]
        if start == None:
            start = 0
        if end == None:
            end = dataset.__len__()
        if resize is None:
            for i in range(start, end):
                self.data.append((*dataset.__getitem__(i)))
        else:
            for i in range(start, end):
                item = dataset.__getitem__(i)
                self.data.append((F.center_crop(F.resize(item[0],resize,Image.BILINEAR), resize), item[1]))
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.transform:
            return (self.transform(self.data[idx][0]), self.data[idx][1], idx)
        else:
            return self.data[idx], idx


class C10(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(C10, self).__init__(root, train=train, transform=transform,
                                     target_transform=target_transform, download=download)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, index

class C100(datasets.CIFAR100):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(C100, self).__init__(root, train=train, transform=transform,
                                     target_transform=target_transform, download=download)

    def __getitem__(self, index):
        # if self.train:
        #     img, target = self.train_data[index], self.train_labels[index]
        # else:
        #     img, target = self.test_data[index], self.test_labels[index]
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

        
class SVHN(datasets.SVHN):
    def __init__(self, root, split='train', transform=None, target_transform=None, download=False):
        super(SVHN, self).__init__(root, split=split, transform=transform,
                                     target_transform=target_transform, download=download)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.labels[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


class CatsDogs(Dataset):
    def __init__(self, root, split='train', transform=None, target_transform=None, resize=None):
        super(CatsDogs, self).__init__()
        self.root = os.path.join(root, "train")
        self.resize  = resize
        self.transform = transform
        self.target_transform = target_transform
        self.data = []
        with open(os.path.join(root, split + "_gambler_split.txt"), 'r') as fin:
            for fname in fin.readlines():
                self.data.append(fname.strip())

    def __getitem__(self, index):
        fname = self.data[index]
        
        # read and scale image
        img = Image.open(os.path.join(self.root, fname))
        if self.resize is not None:
            img = F.center_crop(F.resize(img, self.resize, Image.BILINEAR), self.resize)

        # obtain label
        target = 0 if fname.split('.')[0] == 'cat' else 1

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index
    
    def __len__(self):
        return len(self.data)


from imagenet_classnames import name_map, folder_label_map

# Create a reverse dictionary that maps items to keys
label_folder_map = dict(list(map(lambda z : (z[1], z[0]), folder_label_map.items())))



cur_file_path = os.path.dirname(os.path.abspath(__file__))


with open(os.path.join(cur_file_path, 'imagenet100.txt')) as f:  # The class subset is taken from: https://github.com/HobbitLong/CMC/blob/master/imagenet100.txt
    class_names = list(map(lambda x : x.strip(), f.readlines()))


class ImageNet_Dataset(Dataset): 
    
    def __init__(self, root, transform=None, split=None):
        self.img_path = []
        self.labels = []
        self.transform = transform

        nClasses = 1000 # Subset of ImageNet
        all_class_names = sorted(os.listdir(root))
        for i, name in enumerate(all_class_names):
            folder_name = name
            folder_path = os.path.join(root, folder_name)

            file_names = os.listdir(folder_path)
            if split is not None:
                num_train = int(len(file_names) * 0.8) # 80% Training data
            for j, fid in enumerate(file_names):
                if split == 'train' and j >= num_train: # ensures only the first 80% of data is used for training
                    break
                elif split == 'test' and j < num_train: # skips the first 80% of data used for training
                    continue
                self.img_path.append(os.path.join(folder_path, fid))
                self.labels.append(i)
        print(f"Dataset Size: {len(self.labels)}")
        self.targets = self.labels # Sampler needs to use targets
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, index):

        path = self.img_path[index]
        label = self.labels[index]
        
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        
        if self.transform is not None:
            sample = self.transform(sample)

        return np.asarray(sample), label, index





import pathlib
from typing import Any, Callable, Optional, Tuple

from PIL import Image

from torchvision.datasets.utils import download_and_extract_archive, download_url, verify_str_arg
from torchvision.datasets.vision import VisionDataset




import json
from pathlib import Path
import PIL


