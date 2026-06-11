import torch
from torch.utils.data import Dataset
import numpy as np
from os import listdir
from os.path import isdir

class DopplerDataset(Dataset):
    def __init__(self, dataset_dir, type="train", transform=None, target_transform=None):
        self.dataset_dir      = dataset_dir
        self.list_subdir      = [dir_name for dir_name in listdir(self.dataset_dir) if isdir(self.dataset_dir+'/'+dir_name)]
        self.transform        = transform
        self.target_transform = target_transform

    def __len__(self):
        num_images = 0
        for dir_name in self.list_subdir:
            num_images += len(listdir(self.dataset_dir+'/'+dir_name))
        
        return num_images
    
    def __getitem__(self, idx): # <----  Very slow (?)
        sample = []; label = ""

        current_img_num = 0 
        for dir_name in self.list_subdir:
            current_img_num += len(listdir(self.dataset_dir+'/'+dir_name))
            if idx < current_img_num:
                img_name = sorted(listdir(self.dataset_dir+'/'+dir_name))[idx-current_img_num]

                sample = np.load(self.dataset_dir+'/'+dir_name+'/'+img_name, allow_pickle=True).T
                if self.transform:
                    image = self.transform(image)

                label  = img_name.split("_")[1]
                label  = label[0]                   # Labels like J1, J2 are intended as J 
                if self.target_transform:
                    label = self.target_transform(label)
                break
        
        return sample, label