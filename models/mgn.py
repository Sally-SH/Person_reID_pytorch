import copy
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

from .resnet import resnet50, Bottleneck

class DimReduceLayer(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DimReduceLayer, self).__init__()
        layers = []
        layers.append(
            nn.Conv2d(
                in_channels, out_channels, 1, stride=1, padding=0, bias=False
            )
        )
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        self.layers = nn.Sequential(*layers)
        self._init_reduction(self.layers)

    def _init_reduction(self, reduction):
        # conv
        nn.init.kaiming_normal_(reduction[0].weight, mode='fan_in')

        # bn
        nn.init.normal_(reduction[1].weight, mean=1., std=0.02)
        nn.init.constant_(reduction[1].bias, 0.)

    def forward(self, x):
        return self.layers(x)

class FCLayer(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(FCLayer, self).__init__()
        self.layer = nn.Linear(in_channels, out_channels)
        self._init_fc(self.layer)

    def _init_fc(self, fc):
        nn.init.kaiming_normal_(fc.weight, mode='fan_out')
        nn.init.constant_(fc.bias, 0.)

    def forward(self, x):
        return self.layer(x)

class MGN(nn.Module):
    def __init__(self, num_classes):
        super(MGN, self).__init__()
        num_classes = num_classes
        feats = 256 # number of feature

        resnet = resnet50(pretrained=True)

        self.backone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3[0],
        )

        res_conv4 = nn.Sequential(*resnet.layer3[1:])

        res_g_conv5 = resnet.layer4

        res_p_conv5 = nn.Sequential(
            Bottleneck(1024, 512, downsample=nn.Sequential(nn.Conv2d(1024, 2048, 1, bias=False), nn.BatchNorm2d(2048))),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512))
        res_p_conv5.load_state_dict(resnet.layer4.state_dict())

        self.p1 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_g_conv5))
        self.p2 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))
        self.p3 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))
        
        self.maxpool_zg_p1 = nn.MaxPool2d(kernel_size=(12, 4))
        self.maxpool_zg_p2 = nn.MaxPool2d(kernel_size=(24, 8))
        self.maxpool_zg_p3 = nn.MaxPool2d(kernel_size=(24, 8))
        self.maxpool_zp2 = nn.MaxPool2d(kernel_size=(12, 8))
        self.maxpool_zp3 = nn.MaxPool2d(kernel_size=(8, 8))

        self.reduction_g = DimReduceLayer(2048,feats)
        self.reduction_p2g = DimReduceLayer(2048,feats)
        self.reduction_p20 = DimReduceLayer(2048,feats)
        self.reduction_p21 = DimReduceLayer(2048,feats)
        self.reduction_p3g = DimReduceLayer(2048,feats)
        self.reduction_p30 = DimReduceLayer(2048,feats)
        self.reduction_p31 = DimReduceLayer(2048,feats)
        self.reduction_p32 = DimReduceLayer(2048,feats)

        self.fc_id_2048_0 = FCLayer(feats, num_classes)
        self.fc_id_2048_1 = FCLayer(feats, num_classes)
        self.fc_id_2048_2 = FCLayer(feats, num_classes)

        self.fc_id_256_1_0 = FCLayer(feats, num_classes)
        self.fc_id_256_1_1 = FCLayer(feats, num_classes)
        self.fc_id_256_2_0 = FCLayer(feats, num_classes)
        self.fc_id_256_2_1 = FCLayer(feats, num_classes)
        self.fc_id_256_2_2 = FCLayer(feats, num_classes)
    

    def forward(self, x):

        x = self.backone(x)

        p1 = self.p1(x)
        p2 = self.p2(x)
        p3 = self.p3(x)

        zg_p1 = self.maxpool_zg_p1(p1)
        zg_p2 = self.maxpool_zg_p2(p2)
        zg_p3 = self.maxpool_zg_p3(p3)

        zp2 = self.maxpool_zp2(p2)
        z0_p2 = zp2[:, :, 0:1, :]
        z1_p2 = zp2[:, :, 1:2, :]

        zp3 = self.maxpool_zp3(p3)
        z0_p3 = zp3[:, :, 0:1, :]
        z1_p3 = zp3[:, :, 1:2, :]
        z2_p3 = zp3[:, :, 2:3, :]
        
        fg_p1 = self.reduction_g(zg_p1).squeeze(dim=3).squeeze(dim=2)
        fg_p2 = self.reduction_p2g(zg_p2).squeeze(dim=3).squeeze(dim=2)
        fg_p3 = self.reduction_p3g(zg_p3).squeeze(dim=3).squeeze(dim=2)
        f0_p2 = self.reduction_p20(z0_p2).squeeze(dim=3).squeeze(dim=2)
        f1_p2 = self.reduction_p21(z1_p2).squeeze(dim=3).squeeze(dim=2)
        f0_p3 = self.reduction_p30(z0_p3).squeeze(dim=3).squeeze(dim=2)
        f1_p3 = self.reduction_p31(z1_p3).squeeze(dim=3).squeeze(dim=2)
        f2_p3 = self.reduction_p32(z2_p3).squeeze(dim=3).squeeze(dim=2)

        l_p1 = self.fc_id_2048_0(fg_p1)
        l_p2 = self.fc_id_2048_1(fg_p2)
        l_p3 = self.fc_id_2048_2(fg_p3)
        
        l0_p2 = self.fc_id_256_1_0(f0_p2)
        l1_p2 = self.fc_id_256_1_1(f1_p2)
        l0_p3 = self.fc_id_256_2_0(f0_p3)
        l1_p3 = self.fc_id_256_2_1(f1_p3)
        l2_p3 = self.fc_id_256_2_2(f2_p3)

        predict = torch.cat([fg_p1, fg_p2, fg_p3, f0_p2, f1_p2, f0_p3, f1_p3, f2_p3], dim=1)

        if not self.training:
            return predict

        return fg_p1, fg_p2, fg_p3, l_p1, l_p2, l_p3, l0_p2, l1_p2, l0_p3, l1_p3, l2_p3


def mgn(num_classes):
    model = MGN(num_classes)
    return model

