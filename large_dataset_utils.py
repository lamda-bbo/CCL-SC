
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

from imagenet_classnames import folder_label_map

from torchvision.datasets.stanford_cars import StanfordCars
from torchvision.datasets.food101 import Food101
from torchvision.datasets.celeba import CelebA
# Reverse dictionary mapping
label_folder_map = {v: k for k, v in folder_label_map.items()}

cur_file_path = os.path.dirname(os.path.abspath(__file__))



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


class Celeba(CelebA):
    def __init__(self, root, split='train', target_type='attr', transform=None, download=False):
        super(Celeba, self).__init__(root, split=split, transform=transform, target_type=target_type, download=download)

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target[2], index
