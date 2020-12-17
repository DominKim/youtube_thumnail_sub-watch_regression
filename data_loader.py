# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 14:03:18 2020

@author: User
"""

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pickle
os.environ["KMP_DUPLICATE_LIB_OK"]="True"   


class Img_Dataset(Dataset):
    
    def __init__(self, data, labels):
        super(Img_Dataset, self).__init__()
        self.data = data
        self.labels = labels
        
    def __len__(self):
        return self.data.size(0)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
                  
        return x, y
    
def  Image_get(path, transform):

    lst = []
    for i in path:
        img = Image.open(i)
        img = transform(img)
        lst.append(img)
        
    image = torch.stack(lst)
    pkl_file = open("/Users/mac/Desktop/bigdata/Python/NLP_deep_learning/data/youtube/pkl/youtube_thumnail.pkl", "rb")
    pkl_file = pickle.load(pkl_file)
    labels = pkl_file["watch/sub"].values
    labels = torch.tensor(labels)
    labels = labels.type(torch.float)
    return image, labels
import torch


def get_loaders(config):
    path = ["/Users/mac/Desktop/bigdata/Python/NLP_deep_learning/data/youtube/thumbnail/" + str(i) + ".jpg" for i in range(3200)]
    transform = transforms.Compose([transforms.Resize((224,224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean = 0.5, std = 0.5)])
                
    x, y = Image_get(path, transform)

    train_cnt = int(x.size(0) * config.train_ratio)
    valid_cnt = int(x.size(0) * config.valid_ratio)
    test_cnt = x.size(0) - (train_cnt + valid_cnt)

    # Shuffle dataset to split into train/valid set.
    indices = torch.randperm(x.size(0))
    train_x, valid_x, test_x = torch.index_select(
        x,
        dim=0,
        index=indices
    ).split([train_cnt, valid_cnt, test_cnt], dim=0)
    train_y, valid_y, test_y = torch.index_select(
        y,
        dim=0,
        index=indices
    ).split([train_cnt, valid_cnt, test_cnt], dim=0)

    train_loader = DataLoader(
        dataset=Img_Dataset(train_x, train_y),
        batch_size=config.batch_size,
        # 무조건 shuffle
        shuffle=True,
    )
    valid_loader = DataLoader(
        dataset=Img_Dataset(valid_x, valid_y),
        batch_size=config.batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        dataset = Img_Dataset(test_x, valid_y)
    )

    return train_loader, valid_loader, test_loader

