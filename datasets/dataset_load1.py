# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 12:55:12 2020

@author: 11627
"""
# liver_cancers.py
import os
import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import transforms
from utils import helpers

'''
Background: 0
Liver: 1
Right kidney: 2
Left kidney: 3
Spleen: 4 
'''

palette = [[0], [1], [2], [3], [4]]
num_classes = 5

def make_dataset(root,filelist):

    items = []
    img_path = os.path.join(root, 'imgs1')
    mask_path = os.path.join(root, 'labels1')
    data_list = [l.strip('\n') for l in open(os.path.join(root, filelist)).readlines()]
    for it in data_list:
        item = (os.path.join(img_path, it), os.path.join(mask_path, it))
#        item = os.path.join(img_path, it)
        items.append(item)
    return items

class data_set(data.Dataset):
    def __init__(self, root,filelist):
        self.imgs = make_dataset(root,filelist)
        self.palette = palette
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')

    def __getitem__(self, index):
        img_path = self.imgs[index]
        img_path, mask_path = self.imgs[index]
        img = np.load(img_path)
        
        
        mask = np.load(mask_path)

        img = np.expand_dims(img, axis=2)
        
        
        mask = np.expand_dims(mask, axis=2)
        mask = helpers.mask_to_onehot(mask, self.palette)

        img = img.transpose([2, 0, 1])
        mask = mask.transpose([2, 0, 1])
        
        img = torch.from_numpy(np.array(img))
        mask=torch.from_numpy(np.array(mask, dtype=np.float32))
        return img,mask

    def __len__(self):
        return len(self.imgs)
