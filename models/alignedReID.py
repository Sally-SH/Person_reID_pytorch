from threading import local
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from .resnet import resnet50

class AlignedReID(nn.Module):
    def __init__(self, pretrained, local_conv_out_channels=128, num_classes=None):
        super(AlignedReID, self).__init__()
        self.base = resnet50(pretrained=pretrained)
        planes = 2048
        self.local_conv = nn.Conv2d(
            planes,
            local_conv_out_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.local_bn = nn.BatchNorm2d(local_conv_out_channels)
        self.local_relu = nn.ReLU(inplace=True)

        if num_classes is not None:
            self.fc = nn.Linear(planes, num_classes)
            # init.normal(self.fc.weight, std=0.001)
            # init.constant(self.fc.bias, 0)
        self._init_params()
        
    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, x):
        """
        Returns:
        global_feat: shape [N, C]
        local_feat: shape [N, H, c]
        """
        # shape [N, C, H, W]
        feat = self.base(x)
        # print("resnet50:", feat)
        global_feat = F.avg_pool2d(feat, feat.size()[2:])
        # shape [N, C]
        global_feat = global_feat.view(global_feat.size(0), -1)
        # shape [N, C, H, 1]
        local_feat = torch.mean(feat, -1, keepdim=True)
        local_feat = self.local_relu(self.local_bn(self.local_conv(local_feat)))
        # shape [N, H, c]
        local_feat = local_feat.squeeze(-1).permute(0, 2, 1)

        if not self.training:
            return global_feat
        
        # print("global",global_feat)
        # print("local", local_feat)
        logits = self.fc(global_feat)
        return global_feat, local_feat, logits


def alignedReID(num_classes, pretrained = True):
    model = AlignedReID(pretrained=pretrained, num_classes=num_classes)
    return model