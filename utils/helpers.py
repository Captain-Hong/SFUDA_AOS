# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 13:23:57 2020

@author: 11627
"""

# helpers.py
import os
import csv
import numpy as np


def mask_to_onehot(mask, palette):
    """
    Converts a segmentation mask (H, W, C) to (H, W, K) where the last dim is a one
    hot encoding vector, C is usually 1 or 3, and K is the number of class.
    """
    semantic_map = []
    for colour in palette:
        equality = np.equal(mask, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1).astype(np.float32)
    return semantic_map


def onehot_to_mask(mask, palette):
    """
    Converts a mask (K, H, W) to (C,H, W)
    """

    colour_codes = np.array(palette)
    x = np.uint8(colour_codes[mask.astype(np.uint8)])
    return x
