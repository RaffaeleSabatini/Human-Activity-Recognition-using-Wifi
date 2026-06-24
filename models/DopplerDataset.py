import torch
import numpy as np

from os import listdir
from os.path import isdir
from torch.utils.data import Dataset
from tqdm import tqdm

class DopplerDataset(Dataset):
    def __init__(self, dataset_dir, activities, db_conversion, normalization, transform=None, target_transform=None):
        self.dataset_dir      = dataset_dir
        self.labels_map       = activities
        self.transform        = transform
        self.target_transform = target_transform

        # Loading the full dataset in memory
        self.images_list  = sorted([img_name for img_name in listdir(dataset_dir) if img_name.endswith('.npy') and img_name.startswith('S')])
        self.dataset_size = len(self.images_list)
        self.dataset      = torch.zeros((self.dataset_size, 100, 340), dtype=torch.float32)
        self.labels       = torch.zeros(self.dataset_size, dtype=torch.long)
          
        print(f"Loading dataset {dataset_dir} in memory...")
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

            self.dataset[i] = torch.tensor(sample, dtype=torch.float32) 

            # Label loading
            label          = img_name.split("_")[1][0] # Labels like J1, J2 are intended as J    
            self.labels[i] = torch.tensor(self.labels_map[label], dtype=torch.int32) 
        print(f"Dataset is loaded!\n")

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
        print(f"{"-"*34}\n")