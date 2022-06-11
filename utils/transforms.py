# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 13:26:24 2020

@author: 11627
"""

# transforms.py

import random
import numpy as np
import torch
from PIL import Image, ImageFilter


class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.float32))


class NpyToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.float32))


# 不带归一化
class ImgToTensor(object):
    def __call__(self, img):
        img = torch.from_numpy(np.array(img))
        if isinstance(img, torch.ByteTensor):
            return img.float()