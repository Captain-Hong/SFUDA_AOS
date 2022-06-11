# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 13:32:42 2020

@author: 11627
"""
# loss.py
import torch.nn as nn
import torch
import numpy as np


def diceCoeff(pred, gt, eps=1e-5, activation='sigmoid'):
    r""" computational formula：
        dice = (2 * (pred ∩ gt)) / (pred ∪ gt)
    """

    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = nn.Softmax2d()
    else:
        raise NotImplementedError("Activation implemented for sigmoid and softmax2d")

    pred = activation_fn(pred)

    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)

    intersection = (pred_flat * gt_flat).sum(1)
    unionset = pred_flat.sum(1) + gt_flat.sum(1)
    loss =  (2 * intersection + eps) / (unionset + eps)
#    loss1= torch.nn.functional.cross_entropy(pred, gt)
    return  loss.sum() / N


class SoftDiceLoss(nn.Module):
    __name__ = 'dice_loss'

    def __init__(self, num_classes, activation=None):
        super(SoftDiceLoss, self).__init__()
        self.activation = activation

        self.num_classes = num_classes

    def forward(self, y_pred, y_true):
        class_dice = []

        for i in range(0, self.num_classes):
            class_dice.append(diceCoeff(y_pred[:, i:i + 1, :, :], y_true[:, i:i + 1, :, :], activation=self.activation))
        mean_dice = sum(class_dice)/self.num_classes
        return 1-mean_dice
    
class EntropyLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, prob):
        n, c, h, w = prob.size()
        elementwise_entropy = -prob * torch.log2(prob+ 1e-30)
        if self.reduction == 'none':
            return elementwise_entropy

        sum_entropy = torch.sum(elementwise_entropy, dim=(1,2,3))
        if self.reduction == 'sum':
            
            return sum_entropy/ (h * w * c)

        return torch.mean(sum_entropy)    
    
def entropy_loss(v):
    """
        Entropy loss for probabilistic prediction vectors
        input: batch_size x channels x h x w
        output: batch_size x 1 x h x w
    """
    assert v.dim() == 4
    n, c, h, w = v.size()
    return -torch.sum(torch.mul(v, torch.log2(v + 1e-30))) / (n * h * w * np.log2(c))

def prob_2_entropy(prob):
    """ convert probabilistic prediction maps to weighted self-information maps
    """
    n, c, h, w = prob.size()
    return -torch.mul(prob, torch.log2(prob + 1e-30)) / np.log2(c)


class MaxSquareloss(nn.Module):
    def __init__(self, ignore_index= -1):
        super().__init__()
        self.ignore_index = ignore_index
 
    def forward(self, prob):
        """
        :param pred: predictions (N, C, H, W)
        :param prob: probability of pred (N, C, H, W)
        :return: maximum squares loss
        """
        # prob -= 0.5
        mask = (prob != self.ignore_index)    
        loss = -torch.mean(torch.pow(prob, 2)[mask]) / 2
        return loss











