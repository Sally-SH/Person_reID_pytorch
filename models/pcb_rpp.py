from __future__ import division, absolute_import
import torch
from torch import nn
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

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

    def forward(self, x):
        return self.layers(x)

class PCB(nn.Module):
    """Part-based Convolutional Baseline.
    Reference:
        Sun et al. Beyond Part Models: Person Retrieval with Refined
        Part Pooling (and A Strong Convolutional Baseline). ECCV 2018.
    """

    def __init__(
        self,
        num_classes,
        layers,
        parts=6,
        reduced_dim=256,
    ):

        self.inplanes = 64
        super(PCB, self).__init__()
        self.parts = parts
        self.feature_dim = 512 * Bottleneck.expansion

         # backbone network
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(Bottleneck, 64, layers[0])
        self.layer2 = self._make_layer(Bottleneck, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(Bottleneck, 512, layers[3])

        # pcb layers
        self.parts_avgpool = nn.AdaptiveAvgPool2d((self.parts, 1))
        self.dropout = nn.Dropout(p=0.5)
        self.conv5 = DimReduceLayer(512 * Bottleneck.expansion, reduced_dim)
        self.feature_dim = reduced_dim
        self.classifier = nn.ModuleList(
            [
                nn.Linear(self.feature_dim, num_classes)
                for _ in range(self.parts)
            ]
        )
        _init_params(self.modules())

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def backbone(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
    
    def forward(self, x):
        r = self.backbone(x)
        v_g = self.parts_avgpool(r)

        if not self.training:
            v_g = F.normalize(v_g, p=2, dim=1)
            return v_g.view(v_g.size(0), -1)

        v_g = self.dropout(v_g)
        v_h = self.conv5(v_g)

        y = []
        for i in range(self.parts):
            v_h_i = v_h[:, :, i, :]
            v_h_i = v_h_i.view(v_h_i.size(0),-1)
            y_i = self.classifier[i](v_h_i)
            y.append(y_i)
        
        return y

    def add_rpp(self):
        self.parts_avgpool = RPP(parts=self.parts)
        return self

class RPP(nn.Module):
    def __init__(self, parts=6):
        super(RPP, self).__init__()
        self.parts = parts
        self.feature_dim = 512 * Bottleneck.expansion
        layer1 = []
        layer1.append(nn.Conv2d(
            self.feature_dim,
            self.parts,
            kernel_size=1,
            bias=False
            ))
        self.layer1 = nn.Sequential(*layer1)

        layer2 = []
        layer2.append(nn.BatchNorm2d(self.feature_dim))
        layer2.append(nn.ReLU(inplace=True))
        self.layer2 = nn.Sequential(*layer2)
        self.softmax = nn.Softmax(dim=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        _init_params(self.modules())

    def forward(self, x):
        w = self.layer1(x)
        p = self.softmax(w)
        y = []
        for i in range(self.parts) :
            p_i = p[:, i, :, :]
            p_i = torch.unsqueeze(p_i,1)
            y_i = torch.mul(x, p_i)
            y_i = self.layer2(y_i)
            y_i = self.avgpool(y_i)
            y.append(y_i)
        
        f = torch.cat(y, 2)
        return f


def _init_params(modules):
        for m in modules:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

model_url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth' # resnet50

def init_pretrained_weights(model):
    """Initializes model with pretrained weights.
    
    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """
    pretrain_dict = model_zoo.load_url(model_url)
    model_dict = model.state_dict()
    pretrain_dict = {
        k: v
        for k, v in pretrain_dict.items()
        if k in model_dict and model_dict[k].size() == v.size()
    }
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)

def pcb(num_classes, parts=6, reduced_dim=256, pretrained=True):
    model = PCB(
        num_classes=num_classes,
        layers=[3, 4, 6, 3],
        parts=parts,
        reduced_dim=reduced_dim,
    )
    if pretrained:
        init_pretrained_weights(model)
    return model





