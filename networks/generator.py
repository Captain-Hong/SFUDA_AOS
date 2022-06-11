# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 16:14:30 2021

@author: 11627
"""

import torch
import torch.nn as nn


def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()



class Gene(nn.Module):

    def __init__(self, img_ch=1, num_classes=1):
        super().__init__()


        self.conv1 = nn.Conv2d(in_channels = img_ch, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(in_channels = 64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        
        self.conv3 = nn.Conv2d(in_channels = 32, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.norm3 = nn.BatchNorm2d(16)
        self.relu3 = nn.ReLU()
        
        self.conv4 = nn.Conv2d(in_channels = 16, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.norm4 = nn.BatchNorm2d(8)
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(in_channels = 8, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.norm5 = nn.BatchNorm2d(4)
        self.relu5 = nn.ReLU()

        self.conv6 = nn.Conv2d(in_channels = 4, out_channels=2, kernel_size=3, stride=1, padding=1)
        self.norm6 = nn.BatchNorm2d(2)
        self.relu6 = nn.ReLU()
        
        self.conv7 = nn.Conv2d(in_channels = 2, out_channels=num_classes, kernel_size=1)
        self.norm7 = nn.BatchNorm2d(num_classes)
        self.sig = nn.Sigmoid()
        initialize_weights(self)

    def forward(self,x1):
        x = self.conv1(x1)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.norm4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.norm5(x)
        x = self.relu5(x)
        
        x = self.conv6(x)
        x = self.norm6(x)
        x = self.relu6(x)
        
        x = self.conv7(x)
        x = self.norm7(x)
        x = self.sig(x)
        
        return x

