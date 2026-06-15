import torch
from torch.utils.data import Dataset
import numpy as np
from os import listdir
from os.path import isdir

class DopplerDataset(Dataset):
    def __init__(self, dataset_dir, transform=None, target_transform=None):
        self.dataset_dir      = dataset_dir
        self.images_list      = sorted([img_name for img_name in listdir(self.dataset_dir) if img_name.endswith('.npy') and img_name.startswith('S')])
        self.transform        = transform
        self.target_transform = target_transform
        self.labels_map       = {
            "S": 0,
            "W": 1,
            "R": 2,
            "C": 3,
            "E": 4,
            "L": 5,
            "H": 6,
            "J": 7
        }

    def __len__(self):        
        return len(self.images_list)
    
    def __getitem__(self, idx):
        sample = []; label = ""

        img_name = self.images_list[idx]

        sample = np.load(self.dataset_dir+'/'+img_name, allow_pickle=True).T.copy()
        sample = torch.tensor(sample, dtype=torch.float32)
        if self.transform:
            sample = self.transform(sample)
        image = sample.unsqueeze(0)

        # Labels are extracted by file name and converted to numbers
        label  = img_name.split("_")[1]
        label  = str(label[0])                   # Labels like J1, J2 are intended as J 
        label = torch.tensor(self.labels_map[label], dtype=torch.int32)

        if self.target_transform:
            label = self.target_transform(label)   

        return image, label