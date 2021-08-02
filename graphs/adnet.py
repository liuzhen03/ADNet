#-*- coding:utf-8 _*-  
""" 
@author: LiuZhen
@license: Apache Licence 
@file: adnet.py
@time: 2021/08/02
@contact: liuzhen.pwd@gmail.com
@site:  
@software: PyCharm 

"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dcn_v2 import PCD_Align


class ResidualBlockNoBN(nn.Module):

    def __init__(self, num_feat=64, res_scale=1):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


class make_dilation_dense(nn.Module):

    def __init__(self, nChannels, growthRate, kernel_size=3):
        super(make_dilation_dense, self).__init__()
        self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2 + 1,
                              bias=True, dilation=2)

    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out


class DRDB(nn.Module):

    def __init__(self, nChannels, denseLayer, growthRate):
        super(DRDB, self).__init__()
        num_channels = nChannels
        modules = []
        for i in range(denseLayer):
            modules.append(make_dilation_dense(num_channels, growthRate))
            num_channels += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(num_channels, nChannels, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv_1x1(out)
        out = out + x
        return out


class Pyramid(nn.Module):
    def __init__(self, in_channels=6, n_feats=64):
        super(Pyramid, self).__init__()
        self.in_channels = in_channels
        self.n_feats = n_feats
        num_feat_extra = 1

        self.conv1 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.n_feats, kernel_size=1, stride=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
        layers = []
        for _ in range(num_feat_extra):
            layers.append(ResidualBlockNoBN())
        self.feature_extraction = nn.Sequential(*layers)

        self.downsample1 = nn.Sequential(
            nn.Conv2d(self.n_feats, self.n_feats, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.n_feats, self.n_feats, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
        self.downsample2 = nn.Sequential(
            nn.Conv2d(self.n_feats, self.n_feats, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.n_feats, self.n_feats, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

    def forward(self, x):
        x_in = self.conv1(x)
        x1 = self.feature_extraction(x_in)
        x2 = self.downsample1(x1)
        x3 = self.downsample2(x2)
        return [x1, x2, x3]


class SpatialAttentionModule(nn.Module):

    def __init__(self, n_feats):
        super(SpatialAttentionModule, self).__init__()
        self.att1 = nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size=3, padding=1, bias=True)
        self.att2 = nn.Conv2d(n_feats * 2, n_feats, kernel_size=3, padding=1, bias=True)
        self.relu = nn.LeakyReLU()

    def forward(self, x1, x2):
        f_cat = torch.cat((x1, x2), 1)
        att_map = torch.sigmoid(self.att2(self.relu(self.att1(f_cat))))
        return att_map


class ADNet(nn.Module):

    def __init__(self, nChannel, nDenselayer, nFeat, growthRate, align_version='v0'):
        super(ADNet, self).__init__()
        self.n_channel = nChannel
        self.n_denselayer = nDenselayer
        self.n_feats = nFeat
        self.growth_rate = growthRate
        self.align_version = align_version

        # PCD align module
        self.pyramid_feats = Pyramid(3)
        self.align_module = PCD_Align()

        # Spatial attention module
        self.att_module_l = SpatialAttentionModule(self.n_feats)
        self.att_module_h = SpatialAttentionModule(self.n_feats)

        # feature extraction
        self.feat_exract = nn.Sequential(
            nn.Conv2d(3, nFeat, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )

        # conv1
        self.conv1 = nn.Conv2d(self.n_feats * 6, self.n_feats, kernel_size=3, padding=1, bias=True)
        # 3 x DRDBs
        self.RDB1 = DRDB(self.n_feats, self.n_denselayer, self.growth_rate)
        self.RDB2 = DRDB(self.n_feats, self.n_denselayer, self.growth_rate)
        self.RDB3 = DRDB(self.n_feats, self.n_denselayer, self.growth_rate)
        # conv2
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.n_feats * 3, self.n_feats, kernel_size=1, padding=0, bias=True),
            nn.Conv2d(self.n_feats, self.n_feats, kernel_size=3, padding=1, bias=True)
        )
        # post conv
        self.post_conv = nn.Sequential(
            nn.Conv2d(self.n_feats, self.n_feats, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_feats, 3, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x1, x2, x3):
        x1_t, x1_l = x1[:,0:3,...], x1[:,3:,...]
        x2_t, x2_l = x2[:,0:3,...], x2[:,3:,...]
        x3_t, x3_l = x3[:,0:3,...], x3[:,3:,...]
        # pyramid features of linear domain
        f1_l = self.pyramid_feats(x1_l)
        f2_l = self.pyramid_feats(x2_l)
        f3_l = self.pyramid_feats(x3_l)
        f2_ = f2_l[0]
        # PCD alignment
        f1_aligned_l = self.align_module(f1_l, f2_l)
        f3_aligned_l = self.align_module(f3_l, f2_l)
        # Spatial attention module
        f1_t = self.feat_exract(x1_t)
        f2_t = self.feat_exract(x2_t)
        f3_t = self.feat_exract(x3_t)
        f1_t_A = self.att_module_l(f1_t, f2_t)
        f1_t_ = f1_t * f1_t_A
        f3_t_A = self.att_module_h(f3_t, f2_t)
        f3_t_ = f3_t * f3_t_A

        # fusion subnet
        F_ = torch.cat((f1_aligned_l, f1_t_,  f2_, f2_t, f3_aligned_l, f3_t_), 1)
        F_0 = self.conv1(F_)
        F_1 = self.RDB1(F_0)
        F_2 = self.RDB2(F_1)
        F_3 = self.RDB3(F_2)
        FF = torch.cat((F_1, F_2, F_3), 1)
        FF = self.conv2(FF)
        FF = FF + f2_
        res = self.post_conv(FF)
        return res


def test_model(align_version='v0'):
    from thop import profile
    model = ADNet(6, 5, 64, 32, align_version=align_version).cuda()
    x_1 = torch.from_numpy(np.random.rand(1, 6, 256, 256)).float().cuda()
    x_2 = torch.from_numpy(np.random.rand(1, 6, 256, 256)).float().cuda()
    x_3 = torch.from_numpy(np.random.rand(1, 6, 256, 256)).float().cuda()
    flops, params = profile(model, inputs=(x_1, x_2, x_3), verbose=False)
    print('model(%s): flops: %.3f G, params: %.3f M' % (
    'ADNet', flops / 1000 / 1000 / 1000, params / 1000 / 1000))

if __name__ == '__main__':
    test_model('v0')


