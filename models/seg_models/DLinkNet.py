# python
# -*- coding: utf-8 -*-
# author： HJP
# datetime： 2023/3/10 15:22 
# ide： PyCharm

import torch
import torch.nn as nn
import torchvision.models as models
from models.BaseModel import BaseModel
from module.backbone.resnet import ResNet
import torch.nn.functional as F


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock,self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True)
        )

        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels // 4, n_filters, 1),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.deconv2(x)
        x = self.conv3(x)
        return x


class Dblock_more_dilate(nn.Module):
    def __init__(self, channel):
        super(Dblock_more_dilate, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        self.dilate5 = nn.Conv2d(channel, channel, kernel_size=3, dilation=16, padding=16)
        self.relu = nn.ReLU(inplace=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = self.relu(self.dilate1(x))
        dilate2_out = self.relu(self.dilate2(dilate1_out))
        dilate3_out = self.relu(self.dilate3(dilate2_out))
        dilate4_out = self.relu(self.dilate4(dilate3_out))
        dilate5_out = self.relu(self.dilate5(dilate4_out))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out + dilate5_out
        return out


class Dblock(nn.Module):
    def __init__(self, channel, dilation_series, padding_series):
        super(Dblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=dilation_series[0], padding=padding_series[0])
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=dilation_series[1], padding=padding_series[1])
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=dilation_series[2], padding=padding_series[2])
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=dilation_series[3], padding=padding_series[3])
        self.bn = nn.BatchNorm2d(channel)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.LeakyReLU(inplace=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = self.relu(self.bn(self.dilate1(x)))
        dilate2_out = self.relu(self.bn(self.dilate2(dilate1_out)))
        dilate3_out = self.relu(self.bn(self.dilate3(dilate2_out)))
        dilate4_out = self.relu(self.bn(self.dilate4(dilate3_out)))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out

        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        out = self.sigmoid(out)
        return x * out

class SEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return x * out

class DLinkNet(BaseModel):
    def __init__(self, backbone, n_channels, num_classes, pretrained, se=False):
        super(DLinkNet, self).__init__(backbone, n_channels, num_classes, pretrained)
        self.resnet = ResNet(backbone, pretrained)

        dilation_series = [6, 12, 18, 24]
        padding_series = [6, 12, 18, 24]

        self.ca = ChannelAttention(512)
        self.se = SEModule(512)

        if backbone == 'resnet34':
            filters = [64, 128, 256, 512]
            self.dblock = Dblock(512, dilation_series, padding_series)
        elif backbone == 'resnet50':
            filters = [256, 512, 1024, 2048]
            self.dblock = Dblock_more_dilate(2048)
        else:
            filters = [64, 128, 256, 512]
            self.dblock = Dblock(512, dilation_series, padding_series)

        # 解码器
        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        # self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finaldeconv1 = nn.Sequential(
            nn.ConvTranspose2d(filters[0], 32, 4, 2, 1),
            nn.BatchNorm2d(32)
        )
        # self.finalrelu = nn.ReLU(inplace=True)
        self.finalrelu = nn.LeakyReLU(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        size = x.size()
        e, e1, e2, e3, e4 = self.resnet(x)
        e4 = self.dblock(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu(out)
        out = self.finalconv2(out)
        out = self.finalrelu(out)
        out = self.finalconv3(out)

        out = self.sigmoid(out)
        return out
