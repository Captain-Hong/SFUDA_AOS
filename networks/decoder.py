# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 13:27:23 2020

@author: 11627
"""

# u_net.py
from torch import nn
import torch

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

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Decoder(nn.Module):
    def __init__(self, channels=1):
        super(Decoder, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]


        self.Up5 = up_conv(ch_in=filters[4], ch_out=filters[3])
        self.Up_conv5 = conv_block(ch_in=filters[3], ch_out=filters[3])

        self.Up4 = up_conv(ch_in=filters[3], ch_out=filters[2])
        self.Up_conv4 = conv_block(ch_in=filters[2], ch_out=filters[2])

        self.Up3 = up_conv(ch_in=filters[2], ch_out=filters[1])
        self.Up_conv3 = conv_block(ch_in=filters[1], ch_out=filters[1])

        self.Up2 = up_conv(ch_in=filters[1], ch_out=filters[0])
        self.Up_conv2 = conv_block(ch_in=filters[0], ch_out=filters[0])

        self.Conv_1x1 = nn.Conv2d(filters[0], channels, kernel_size=1, stride=1, padding=0)
        self.active = torch.nn.Sigmoid()
        initialize_weights(self)

    def forward(self, x5):

        # decoding
        d5 = self.Up5(x5)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        d1 = self.active(d1)

        return d1
    
    
backbone = 'resnet34'

class DecoderBlock(nn.Module):
    """
    U-Net中的解码模块
    采用每个模块一个stride为1的3*3卷积加一个上采样层的形式
    上采样层可使用'deconv'、'pixelshuffle', 其中pixelshuffle必须要mid_channels=4*out_channles
    定稿采用pixelshuffle
    BN_enable控制是否存在BN，定稿设置为True
    """
    def __init__(self, in_channels, mid_channels, out_channels, upsample_mode='pixelshuffle'):
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.upsample_mode = upsample_mode

    
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1, padding=1, bias=False)

 
        self.norm1 = nn.BatchNorm2d(mid_channels)
        self.relu1 = nn.ReLU(inplace=False)
        self.relu2 = nn.ReLU(inplace=False)

        if self.upsample_mode=='deconv':
            self.upsample = nn.ConvTranspose2d(in_channels=mid_channels, out_channels = out_channels,

                                                kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        elif self.upsample_mode=='pixelshuffle':
            self.upsample = nn.PixelShuffle(upscale_factor=2)

        self.norm2 = nn.BatchNorm2d(out_channels)

    def forward(self,x):
        x=self.conv(x)
        x=self.norm1(x)
        x=self.relu1(x)
        x=self.upsample(x)
        x=self.norm2(x)
        x=self.relu2(x)
        return x

class Resnet_Decoder(nn.Module):
    """
    定稿使用resnet50作为backbone
    BN_enable控制是否存在BN，定稿设置为True
    """
    def __init__(self,resnet_pretrain=False,channels=1):
        super().__init__()

        # encoder部分
        # 使用resnet34或50预定义模型，由于单通道入，因此自定义第一个conv层，同时去掉原fc层
        # 剩余网络各部分依次继承
        # 经过测试encoder取三层效果比四层更佳，因此降采样、升采样各取4次
        if backbone=='resnet34':
            filters=[64,64,128,256,512]
        elif backbone=='resnet50':
            filters=[64,256,512,1024,2048]

        # decoder部分
        self.center = DecoderBlock(in_channels=filters[3], mid_channels=filters[3]*4, out_channels=filters[3])
        self.decoder1 = DecoderBlock(in_channels=filters[3], mid_channels=filters[2]*4, out_channels=filters[2])
        self.decoder2 = DecoderBlock(in_channels=filters[2], mid_channels=filters[1]*4, out_channels=filters[1])
        self.decoder3 = DecoderBlock(in_channels=filters[1], mid_channels=filters[0]*4, out_channels=filters[0])

        self.final = nn.Sequential(
            nn.Conv2d(in_channels=filters[0],out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), 
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=32, out_channels=channels, kernel_size=1),
            nn.Sigmoid()
            )


    def forward(self,x):


        center = self.center(x)

        d2 = self.decoder1(center)
        d3 = self.decoder2(d2)
        d4 = self.decoder3(d3)
        out= self.final(d4)
        return out 