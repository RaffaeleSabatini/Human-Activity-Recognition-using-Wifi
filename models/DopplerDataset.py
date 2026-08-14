import torch
import numpy as np

import torchvision.transforms.functional as F
from os import listdir
from os.path import isdir
from torch.utils.data import Dataset, TensorDataset
from tqdm import tqdm

class DopplerDataset(Dataset):
    def __init__(self, dataset_dir, activities, db_conversion, normalization, augmentation=False, transform=None, target_transform=None):
        self.dataset_dir      = dataset_dir
        self.labels_map       = activities
        self.transform        = transform
        self.target_transform = target_transform
        self.augmentation     = augmentation

        # Loading the full dataset in memory
        self.images_list  = sorted([img_name for img_name in listdir(dataset_dir) if img_name.endswith('.npy') and img_name.startswith('S')])
        size_multiplier = 2 if augmentation in ["h", "v"] else 3 if augmentation == "hv" else 1
        self.dataset_size = len(self.images_list)*size_multiplier
        self.dataset      = torch.zeros((self.dataset_size, 100, 340), dtype=torch.float32)
        self.labels       = torch.zeros(self.dataset_size, dtype=torch.long)
          
        print(f"Loading dataset {dataset_dir} in memory...")
        j = 0
        for i, img_name in tqdm(enumerate(self.images_list)):
            # Image pre-processing 
            sample = np.load(dataset_dir+'/'+img_name, allow_pickle=True).T.copy()

            if db_conversion and normalization:
                print("Warning: conversion to db and normalization may lead to underperforming model...")

            if db_conversion:
                sample = 10*np.log10(sample) 
                sample = sample-sample.max()
            if normalization:
                sample = (sample - sample.mean())/sample.std()
            
            # Image and label loading
            self.dataset[j] = torch.tensor(sample, dtype=torch.float32)
            label          = img_name.split("_")[1][0] # Labels like J1, J2 are intended as J   
            self.labels[j] = torch.tensor(self.labels_map[label], dtype=torch.int32)

            if self.augmentation == "h":
                self.dataset[j+1] = F.hflip(self.dataset[j])
                self.labels[j+1] = torch.tensor(self.labels_map[label], dtype=torch.int32)
                j += 2
            elif self.augmentation == "v":
                self.dataset[j+1] = F.vflip(self.dataset[j])
                self.labels[j+1] = torch.tensor(self.labels_map[label], dtype=torch.int32)
                j += 2
            elif self.augmentation == "hv":
                self.dataset[j+1] = F.hflip(self.dataset[j])
                self.dataset[j+2] = F.vflip(self.dataset[j])
                self.labels[j+1] = torch.tensor(self.labels_map[label], dtype=torch.int32)
                self.labels[j+2] = torch.tensor(self.labels_map[label], dtype=torch.int32)
                j += 3
            else:
                j += 1
            
        print(f"Dataset is loaded!")

    def __len__(self):        
        return self.dataset_size
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        if self.transform:
            sample = self.transform(sample)
        sample = sample.unsqueeze(0)

        label = self.labels[idx]
        if self.target_transform:
            label = self.target_transform(label)   

        return sample, label
    
    def getInfo(self):
        print(f"\n{"-"*10} DATASET INFO {"-"*10}")
        print(f"Dataset directory:\n{self.dataset_dir}")
        print(f"Labels dictionary:\n{self.labels_map}")
        print(f"Dataset size: {self.dataset_size}")
        print(f"{"-"*34}\n")

    def retrieve_activity(self, label):
        mask = self.labels == label
        X_tensor = self.dataset[mask]
        Y_tensor = self.labels[mask]
        return TensorDataset(X_tensor, Y_tensor, ) 