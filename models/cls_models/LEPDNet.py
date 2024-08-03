# python
# -*- coding: utf-8 -*-
# author： HJP
# datetime： 2023/3/21 9:46 
# ide： PyCharm

from models.BaseModel import BaseModel
from module.backbone.resnet import ResNet
from models.seg_models.DLinkNet import DecoderBlock, Dblock_more_dilate, Dblock

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


res = {
    'resnet101': models.resnet101,
    'resnet50': models.resnet50,
    'resnet18': models.resnet18,
    'resnet34': models.resnet34,
    'resnet152': models.resnet152,
}


class DLinkNet(BaseModel):
    def __init__(self, backbone, n_channels, num_classes, pretrained):
        super(DLinkNet, self).__init__(backbone, n_channels, num_classes, pretrained)
        self.resnet = ResNet(backbone, pretrained)

        dilation_series = [6, 12, 18, 24]
        padding_series = [6, 12, 18, 24]

        if backbone == 'resnet34' or 'resnet18':
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
        e, e1, e2, e3, l4 = self.resnet(x)
        e4 = self.dblock(l4)

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
        return out, e4
        # return out, l4

class LocationEmbedding(nn.Module):
    def __init__(self, in_planes=6, out_planes=64):
        super(LocationEmbedding, self).__init__()

        self.conv = nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm1d(out_planes)
        self.silu = nn.SiLU(inplace=True)
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        return self.silu(self.bn(self.conv(x)))

# # Transformer module
class ImagePairCrossAttention(nn.Module):
    """ Image Pair Cross Attention Layer"""
    def __init__(self, in_dim, dropout=0.1, layer_norm_eps=1e-5):
        super().__init__()
        self.in_dim = in_dim

        self.self_attn = nn.MultiheadAttention(in_dim, 1, batch_first=True)

        self.norm1 = nn.LayerNorm(in_dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(in_dim, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self._reset_parameters()

    def forward(self, x, y):
        x = x.view(x.shape[0], self.in_dim, -1).permute(0, 2, 1)
        y = y.view(y.shape[0], self.in_dim, -1).permute(0, 2, 1)

        weight = self._sa_block(self.norm1(x), self.norm1(y))

        return weight.diagonal(dim1=1, dim2=2)

    def _sa_block(self, x, y):
        x, weight = self.self_attn(x, y, y, need_weights=True)
        return weight

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class ContextAwareRegionAttention(nn.Module):
    def __init__(self, in_dim, dropout=0.1, layer_norm_eps=1e-5):
        super().__init__()
        self.in_dim = in_dim
        dim_feedforward = self.in_dim * 4

        self.self_attn = nn.MultiheadAttention(in_dim, 8, batch_first=True)

        self.linear1 = nn.Linear(in_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, in_dim)

        self.norm1 = nn.LayerNorm(in_dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(in_dim, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.relu
        self._reset_parameters()

        self.classifier = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, 1, bias=False),
        )

    def forward(self, x, y):
        b, c, h, w = x.shape
        x = x.view(x.shape[0], self.in_dim, -1).permute(0, 2, 1)
        y = y.view(y.shape[0], self.in_dim, -1).permute(0, 2, 1)

        y = self._sa_block(self.norm1(x), self.norm1(y))
        y = y.permute(0, 2, 1).view(b, c, h, w)

        return y

    def _sa_block(self, x, y):
        x = self.self_attn(x, y, y, need_weights=False)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

class StoneLocationEmbeddingModule(nn.Module):
    def __init__(self, in_channels=576):
        super(StoneLocationEmbeddingModule, self).__init__()
        self.text_net = LocationEmbedding()
        self.fusion_module = nn.Sequential(
            nn.Conv2d(in_channels, 768, 1, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
            nn.Conv2d(768, 512, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, text, x=None):
        if text is not None:
            text = text.unsqueeze(-1)
            text_feature = self.text_net(text).unsqueeze(-1).expand(-1, 64, 7, 7)
            out = self.fusion_module(torch.cat([x, text_feature], dim=1))
        else:
            out = x
        return out


class LEPDNet(BaseModel):
    def __init__(self, backbone, n_channels, num_classes, pretrained):
        super(LEPDNet, self).__init__(backbone, n_channels, num_classes, pretrained)
        self.backbone = ResNet(backbone, pretrained=pretrained)
        self.seg_net = DLinkNet(backbone, 3, 1, pretrained)

        self.CARA = ContextAwareRegionAttention(512)
        self.IPCA = ImagePairCrossAttention(512)
        self.SLE = StoneLocationEmbeddingModule()  # backbone last channels + text channels(64)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        self.m = 0.999
        self.T = 0.07
        dim = 128

        self.fc1 = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, dim)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, dim)
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(512, 1024, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, 1, bias=False),
        )


        self.feats = None
        self.backbone_parameters = list(self.backbone.parameters())

    def get_backbone_layers(self):
        small_lr_layers = []
        small_lr_layers.append(self.seg_net)
        return small_lr_layers


    def get_feats(self):
        feats = self.feats
        return feats

    def forward(self, x, seg_hx=None, text=None, pair_x=None, pair_text=None, mode='cls', seg_out=None):
        '''
        seg_hx: 分割网络的高阶特征
        text: 文本信息
        '''
        if mode == 'seg':
            seg_out, seg_hx = self.seg_net(x)
            return seg_out, seg_hx
        elif mode == 'cls':
            _, l1, l2, l3, l4 = self.backbone(x)
            if seg_hx is not None:
                fusion_l4 = self.fusion(l4 + self.CARA(seg_hx, l4))
            else:
                fusion_l4 = l4

            out = self.SLE(text, fusion_l4)

            out = self.avgpool(out)
            out = torch.flatten(out, 1)
            self.feats = out.detach().clone()
            out = self.fc(out)
            return out
        elif mode == 'bicls':
            _, l1, l2, l3, l4 = self.backbone(x)

            if seg_hx is not None:
                fusion_l4 = self.fusion(l4 + self.CARA(seg_hx, l4))
            else:
                fusion_l4 = l4

            out1 = self.SLE(text, fusion_l4)

            with torch.no_grad():
                _, pair_seg_hx = self.seg_net(pair_x)

            _, l1, l2, l3, pair_l4 = self.backbone(pair_x)
            pair_fusion_l4 = self.fusion(pair_l4 + self.CARA(pair_seg_hx, pair_l4))
            pair_out1 = self.SLE(pair_text, pair_fusion_l4)

            w2 = self.IPCA(pair_out1, out1)

            w1 = self.IPCA(out1, pair_out1.detach())

            out = self.avgpool(out1)
            out = torch.flatten(out, 1)
            out = self.fc(out)
            return out, w1, w2