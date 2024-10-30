import torch
import math
import torch
import torch.nn as nn

import torch.nn.functional as F
from torch.nn import Conv2d, Module, Linear, BatchNorm2d, ReLU
from torch.nn.modules.utils import _pair



class SplAtConv2d(Module):
    """Split-Attention Conv2d
    """
    def __init__(self, in_channels, channels, kernel_size, stride=(1, 1), padding=(0, 0),
                 dilation=(1, 1), groups=1, bias=True,
                 radix=2, reduction_factor=4,
                 rectify=False, rectify_avg=False, norm_layer=None,
                 dropblock_prob=0.0, **kwargs):
        super(SplAtConv2d, self).__init__()
        padding = _pair(padding)
        self.rectify = rectify and (padding[0] > 0 or padding[1] > 0)
        self.rectify_avg = rectify_avg
        inter_channels = max(in_channels*radix//reduction_factor, 32)
        self.radix = radix
        self.cardinality = groups
        self.channels = channels
        self.dropblock_prob = dropblock_prob
        if self.rectify:
            from rfconv import RFConv2d
            self.conv = RFConv2d(in_channels, channels*radix, kernel_size, stride, padding, dilation,
                                 groups=groups*radix, bias=bias, average_mode=rectify_avg, **kwargs)
        else:
            self.conv = Conv2d(in_channels, channels*radix, kernel_size, stride, padding, dilation,
                               groups=groups*radix, bias=bias, **kwargs)
        self.use_bn = norm_layer is not None
        if self.use_bn:
            self.bn0 = norm_layer(channels*radix)
        self.relu = ReLU(inplace=True)
        self.fc1 = Conv2d(channels, inter_channels, 1, groups=self.cardinality)
        if self.use_bn:
            self.bn1 = norm_layer(inter_channels)
        self.fc2 = Conv2d(inter_channels, channels*radix, 1, groups=self.cardinality)
        if dropblock_prob > 0.0:
            self.dropblock = DropBlock2D(dropblock_prob, 3)
        self.rsoftmax = rSoftMax(radix, groups)


        self.c1 = nn.Sequential(Conv2d(in_channels, channels, 1, stride, 0, dilation, bias=bias),
                                norm_layer(channels), ReLU(inplace=True))
        self.c2 = nn.Sequential(Conv2d(in_channels, channels, 3, stride, 1, dilation, bias=bias),
                                norm_layer(channels), ReLU(inplace=True))
        self.c3 = nn.Sequential(Conv2d(in_channels, channels, 5, stride, 2, dilation, bias=bias),
                                norm_layer(channels), ReLU(inplace=True))
        self.c4 = nn.Sequential(Conv2d(in_channels, channels, 7, stride, 3, dilation, bias=bias),
                                norm_layer(channels), ReLU(inplace=True))
    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn0(x)
        if self.dropblock_prob > 0.0:
            x = self.dropblock(x)
        x = self.relu(x)

        batch, rchannel = x.shape[:2]
                               
        x1, x2, x3, x4 = torch.split(x, rchannel//self.radix, dim=1)
        x1=self.c1(x1);x2=self.c2(x2);x3=self.c3(x3);x4=self.c4(x4);
        splited = (x1,x2,x3,x4)
        
        out = sum([split for split in splited])

        return out.contiguous()

class rSoftMax(nn.Module):
    def __init__(self, radix, cardinality):
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x

class DropBlock2D(object):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        return nn.functional.adaptive_avg_pool2d(inputs, 1).view(inputs.size(0), -1)

class Bottleneck(nn.Module):
    """ResNet Bottleneck
    """
    # pylint: disable=unused-argument
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 radix=1, cardinality=1, bottleneck_width=64,
                 avd=False, avd_first=False, dilation=1, is_first=False,
                 rectified_conv=False, rectify_avg=False,
                 norm_layer=None, dropblock_prob=0.0, last_gamma=False):
        super(Bottleneck, self).__init__()
        group_width = int(planes * (bottleneck_width / 64.)) * cardinality
        self.conv1 = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(group_width)
        self.dropblock_prob = dropblock_prob
        self.radix = radix
        self.avd = avd and (stride > 1 or is_first)
        self.avd_first = avd_first

        if self.avd:
            self.avd_layer = nn.AvgPool2d(3, stride, padding=1)
            stride = 1

        if dropblock_prob > 0.0:
            self.dropblock1 = DropBlock2D(dropblock_prob, 3)
            if radix == 1:
                self.dropblock2 = DropBlock2D(dropblock_prob, 3)
            self.dropblock3 = DropBlock2D(dropblock_prob, 3)

        if radix >= 1:
            self.conv2 = SplAtConv2d(
                group_width, group_width, kernel_size=3,
                stride=stride, padding=dilation,
                dilation=dilation, groups=cardinality, bias=False,
                radix=radix, rectify=rectified_conv,
                rectify_avg=rectify_avg,
                norm_layer=norm_layer,
                dropblock_prob=dropblock_prob)
        elif rectified_conv:
            from rfconv import RFConv2d
            self.conv2 = RFConv2d(
                group_width, group_width, kernel_size=3, stride=stride,
                padding=dilation, dilation=dilation,
                groups=cardinality, bias=False,
                average_mode=rectify_avg)
            self.bn2 = norm_layer(group_width)
        else:
            self.conv2 = nn.Conv2d(
                group_width, group_width, kernel_size=3, stride=stride,
                padding=dilation, dilation=dilation,
                groups=cardinality, bias=False)
            self.bn2 = norm_layer(group_width)

        self.conv3 = nn.Conv2d(
            group_width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes*4)

        if last_gamma:
            from torch.nn.init import zeros_
            zeros_(self.bn3.weight)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        if self.dropblock_prob > 0.0:
            out = self.dropblock1(out)
        out = self.relu(out)

        if self.avd and self.avd_first:
            out = self.avd_layer(out)

        out = self.conv2(out)
        if self.radix == 0:
            out = self.bn2(out)
            if self.dropblock_prob > 0.0:
                out = self.dropblock2(out)
            out = self.relu(out)

        if self.avd and not self.avd_first:
            out = self.avd_layer(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.dropblock_prob > 0.0:
            out = self.dropblock3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, in_channles, num_classes, block, layers, radix=1, groups=1, bottleneck_width=64,
                 dilated=False, dilation=1,
                 deep_stem=False, stem_width=64, avg_down=False,
                 rectified_conv=False, rectify_avg=False,
                 avd=False, avd_first=False,
                 final_drop=0.0, dropblock_prob=0,
                 last_gamma=False, norm_layer=nn.BatchNorm2d):
        self.cardinality = groups
        self.bottleneck_width = bottleneck_width
        # ResNet-D params
        self.inplanes = stem_width*2 if deep_stem else 64
        self.avg_down = avg_down
        self.last_gamma = last_gamma
        # ResNeSt params
        self.radix = radix
        self.avd = avd
        self.avd_first = avd_first

        super(ResNet, self).__init__()
        self.rectified_conv = rectified_conv
        self.rectify_avg = rectify_avg
        if rectified_conv:
            from rfconv import RFConv2d
            conv_layer = RFConv2d
        else:
            conv_layer = nn.Conv2d
        conv_kwargs = {'average_mode': rectify_avg} if rectified_conv else {}

        self.conv1 = conv_layer(3, 64, kernel_size=7, stride=2, padding=3,
                                   bias=False, **conv_kwargs)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 32, layers[0], norm_layer=norm_layer, is_first=False)
        self.layer2 = self._make_layer(block, 64, layers[1], stride=1, norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=1,
                                           norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob)
        self.layer4 = self._make_layer(block, 128, layers[3], stride=1,
                                           norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob)
        
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, norm_layer):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=None,
                    dropblock_prob=0.0, is_first=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            down_layers = []
            down_layers.append(nn.Conv2d(self.inplanes, planes * block.expansion,
                                             kernel_size=1, stride=stride, bias=False))
            down_layers.append(norm_layer(planes * block.expansion))
            downsample = nn.Sequential(*down_layers)

        layers = []

        layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                                radix=self.radix, cardinality=self.cardinality,
                                bottleneck_width=self.bottleneck_width,
                                avd=self.avd, avd_first=self.avd_first,
                                dilation=1, is_first=is_first, rectified_conv=self.rectified_conv,
                                rectify_avg=self.rectify_avg,
                                norm_layer=norm_layer, dropblock_prob=dropblock_prob,
                                last_gamma=self.last_gamma))
       

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                radix=self.radix, cardinality=self.cardinality,
                                bottleneck_width=self.bottleneck_width,
                                avd=self.avd, avd_first=self.avd_first,
                                dilation=dilation, rectified_conv=self.rectified_conv,
                                rectify_avg=self.rectify_avg,
                                norm_layer=norm_layer, dropblock_prob=dropblock_prob,
                                last_gamma=self.last_gamma))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        #x = x.view(x.size(0), -1)
        x = torch.flatten(x, 1)
        if self.drop:
            x = self.drop(x)
        x = self.fc(x)

        return x


    

class HSNet(nn.Module):
    def __init__(self,in_channels=3,num_classes=1, init_features=64):
        super(HSNet, self).__init__()

        batchNorm_momentum = 0.1
        num_features = init_features
        resnest = ResNet(1, 1, Bottleneck, [1, 1, 1, 1],
                   radix=4, groups=1, bottleneck_width=64,
                   deep_stem=False, stem_width=32, avg_down=False,
                   avd=False, avd_first=False)
        
        # Encoder
        self.conv11 = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(num_features, momentum= batchNorm_momentum)
        self.conv12 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(num_features, momentum= batchNorm_momentum)
        self.conv13 = nn.Conv2d(num_features, num_features, kernel_size=1, padding=0)
        self.bn13 = nn.BatchNorm2d(num_features, momentum= batchNorm_momentum)
        self.down1 = resnest.layer1
        self.down2 = resnest.layer2
        self.down3 = resnest.layer3
        self.down4 = resnest.layer4
        
        self.conv43d = nn.Conv2d(num_features*8*2, num_features*8, kernel_size=3, padding=1)
        self.bn43d = nn.BatchNorm2d(num_features*8)
        self.conv42d = nn.Conv2d(num_features*8, num_features*8, kernel_size=3, padding=1)
        self.bn42d = nn.BatchNorm2d(num_features*8)
        self.conv41d = nn.Conv2d(num_features*8, num_features*4, kernel_size=3, padding=1)
        self.bn41d = nn.BatchNorm2d(num_features*4)
        self.conv33d = nn.Conv2d(num_features*4*2, num_features*4, kernel_size=3, padding=1)
        self.bn33d = nn.BatchNorm2d(num_features*4)
        self.conv32d = nn.Conv2d(num_features*4, num_features*4, kernel_size=3, padding=1)
        self.bn32d = nn.BatchNorm2d(num_features*4)
        self.conv31d = nn.Conv2d(num_features*4,  num_features*2, kernel_size=3, padding=1)
        self.bn31d = nn.BatchNorm2d(num_features*2)
        self.conv22d = nn.Conv2d(num_features*2*2, num_features*2, kernel_size=3, padding=1)
        self.bn22d = nn.BatchNorm2d(num_features*2)
        self.conv21d = nn.Conv2d(num_features*2, num_features, kernel_size=3, padding=1)
        self.bn21d = nn.BatchNorm2d(num_features)
        self.conv12d = nn.Conv2d(num_features*2, num_features*2, kernel_size=3, padding=1)
        self.bn12d = nn.BatchNorm2d(num_features*2)
        self.conv11d = nn.Conv2d(num_features*2, num_classes, kernel_size=3, padding=1)

    def forward(self, x):
        x11 = F.relu(self.bn11(self.conv11(x)))
        x12 = F.relu(self.bn12(self.conv12(x11)))
        x13 = F.relu(self.bn13(self.conv13(x12)))
        x1p, id1 = F.max_pool2d(x13, kernel_size=2, stride=2,return_indices=True)
        
        x22 = self.down1(x1p)
        x2p, id2 = F.max_pool2d(x22,kernel_size=2, stride=2,return_indices=True)

        x33 = self.down2(x2p)
        x3p, id3 = F.max_pool2d(x33,kernel_size=2, stride=2,return_indices=True)

        x43 = self.down3(x3p)
        x4p, id4 = F.max_pool2d(x43,kernel_size=2, stride=2,return_indices=True)
        
        x53 = self.down4(x4p)
        
        x4d = F.max_unpool2d(x53, id4, kernel_size=2, stride=2)
        x4d = torch.cat((x4d, x43), dim=1)
        x43d = F.relu(self.bn43d(self.conv43d(x4d)))
        x42d = F.relu(self.bn42d(self.conv42d(x43d)))
        x41d = F.relu(self.bn41d(self.conv41d(x42d)))
        x3d = F.max_unpool2d(x41d, id3, kernel_size=2, stride=2)
        x3d = torch.cat((x3d, x33), dim=1)
        x33d = F.relu(self.bn33d(self.conv33d(x3d)))
        x32d = F.relu(self.bn32d(self.conv32d(x33d)))
        x31d = F.relu(self.bn31d(self.conv31d(x32d)))
        x2d = F.max_unpool2d(x31d, id2, kernel_size=2, stride=2)
        x2d = torch.cat((x2d, x22), dim=1)
        x22d = F.relu(self.bn22d(self.conv22d(x2d)))
        x21d = F.relu(self.bn21d(self.conv21d(x22d)))
        x1d = F.max_unpool2d(x21d, id1, kernel_size=2, stride=2)
        x1d = torch.cat((x1d, x13), dim=1)
        x1d = F.relu(self.bn12d(self.conv12d(x1d)))     
        x11d = self.conv11d(x1d)

        return x11d

# import pytorch_model_summary
# print(pytorch_model_summary.summary(HSNet(1),torch.rand((1, 1, 512, 512))))   
