import logging
import math

import torch
import torch.nn as nn
from fds import FDS
from prm import PRM
from evidential_deep_learning import *

print = logging.info


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class Calibration_model(nn.Module):
    def __init__(self):
        super(Calibration_model, self).__init__()

        self.layer1 = nn.Linear(1, 1)
        self.layer2 = nn.Linear(1, 1)
        self.layer3 = nn.Linear(1, 1)

    def forward(self, x):
        mu, nu, al, be = torch.tensor_split(x, 4, dim=1)
        nu_new = self.layer1(nu)
        al_new = self.layer2(al) + 1.0
        be_new = self.layer3(be)

        return torch.cat([mu, nu_new, al_new, be_new], dim=1)

    def decouple(self, w):
        return w.data.cpu().numpy()

    def get_weights(self):
        return self.decouple(self.layer1.weight), self.decouple(self.layer2.weight), self.decouple(self.layer3.weight)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, is_last=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.is_last = is_last
        if is_last:
            self.edl = Conv2DNormal(planes, planes * 4, 1, bias=False)
            self.edl_bn3 = nn.BatchNorm2d(planes * 4)
        else:
            self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
            self.bn3 = nn.BatchNorm2d(planes * 4)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        if self.is_last:
            out = self.edl(out)
            d = int(out.size(1) / 2)
            out[:, :d, :, :] = self.edl_bn3(out[:, :d, :, :])
        else:
            out = self.conv3(out)
            out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.is_last:
            out[:, :d, :, :] += residual
        else:
            out += residual
        out = self.relu(out)
        return out


class ReverseBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ReverseBottleneck, self).__init__()
        self.inplanes = inplanes
        self.planes = planes

        self.reverse_conv1 = nn.ConvTranspose2d(planes, inplanes, kernel_size=1, bias=False)
        self.reverse_bn1 = nn.BatchNorm2d(planes)
        if stride == 1:
            self.reverse_conv2 = nn.ConvTranspose2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        else:
            self.reverse_conv2 = nn.ConvTranspose2d(planes, planes, kernel_size=2, stride=stride, padding=0, bias=False)
        self.reverse_bn2 = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.reverse_conv3 = nn.ConvTranspose2d(planes * 4, planes, kernel_size=1, bias=False)
        self.reverse_bn3 = nn.BatchNorm2d(planes * 4)

    def forward(self, x):
        out = self.relu(x)
        out = self.reverse_bn3(out)
        out = self.reverse_conv3(out)
        out = self.relu(out)
        out = self.reverse_bn2(out)
        out = self.reverse_conv2(out)
        out = self.relu(out)
        out = self.reverse_bn1(out)
        out = self.reverse_conv1(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, reverse_block, layers, fds, bucket_num, bucket_start, start_update, start_smooth,
                 kernel, ks, sigma, momentum, dropout=None, use_edl=False, use_cdm=False, use_prm=False,
                 use_recons=False, bins=1, device=None):
        self.inplanes = 64

        self.split_size = 512 * block.expansion
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, is_last=use_prm)

        if use_recons:
            self.reverse_layer1 = self._make_layer(reverse_block, 64, layers[0], is_reverse=True, is_first=True)
            self.reverse_layer2 = self._make_layer(reverse_block, 128, layers[1], stride=2, is_reverse=True)
            self.reverse_layer3 = self._make_layer(reverse_block, 256, layers[2], stride=2, is_reverse=True)
            self.reverse_layer4 = self._make_layer(reverse_block, 512, layers[3], stride=2, is_reverse=True)

            self.reverse_initial = nn.Sequential(
                nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2, padding=0, bias=False),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(64),
                nn.ConvTranspose2d(64, 3, kernel_size=6, stride=2, padding=2, bias=False))
            self.reverse_avgpool = nn.ConvTranspose2d(2048, 2048, kernel_size=7, stride=7, padding=0, bias=False)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.linear = nn.Linear(self.split_size, 1)

        self.linear_normal_edl = LinearNormalGamma(self.split_size, 1).to(device=device)
        self.conjugate_prior = ConjugatePrior(self.split_size, 1).to(device=device)

        self.infer_pos = -1

        if fds:
            if use_prm:
                print('use PRM')
                self.FDS = PRM(
                    feature_dim=self.split_size, bucket_num=bucket_num, bucket_start=bucket_start,
                    start_update=start_update, start_smooth=start_smooth, kernel=kernel, ks=ks, sigma=sigma,
                    momentum=momentum, bins=bins, device=device
                ).to(device=device)
            else:
                print('use FDS')
                self.FDS = FDS(
                    feature_dim=self.split_size, bucket_num=bucket_num, bucket_start=bucket_start,
                    start_update=start_update, start_smooth=start_smooth, kernel=kernel, ks=ks, sigma=sigma,
                    momentum=momentum, bins=bins, device=device
                ).to(device=device)

        self.fds = fds
        self.start_smooth = start_smooth

        self.use_dropout = True if dropout else False
        self.use_edl = use_edl
        self.use_cdm = use_cdm
        self.use_prm = use_prm
        self.use_recons = use_recons

        if self.use_dropout:
            print(f'Using dropout: {dropout}')
            self.dropout = nn.Dropout(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, is_reverse=False, is_last=False, is_first=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if not is_reverse:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )
        layers = []

        if is_reverse:
            layers.append(block(planes * 4, planes))
            for i in range(1, blocks):

                if i == blocks - 1:
                    if is_first:
                        layers.append(block(planes, planes))
                    else:
                        layers.append(block(planes * 2, planes, stride=stride))

                else:
                    layers.append(block(planes * 4, planes))
        else:
            layers.append(block(self.inplanes, planes, stride, downsample))
            self.inplanes = planes * block.expansion
            for i in range(1, blocks):

                if i == blocks - 1:
                    layers.append(block(self.inplanes, planes, is_last=is_last))
                else:
                    layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def set_infernoise(self, pos):
        self.infer_pos = pos

    def forward(self, x, targets=None, epoch=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        if not self.training and self.infer_pos == 1:
            x += torch.empty_like(x).normal_(0, 50.0)
        x = self.layer2(x)
        if not self.training and self.infer_pos == 2:
            x += torch.empty_like(x).normal_(0, 50.0)
        x = self.layer3(x)
        if not self.training and self.infer_pos == 3:
            x += torch.empty_like(x).normal_(0, 50.0)
        x = self.layer4(x)
        if not self.training and self.infer_pos == 4:
            x += torch.empty_like(x).normal_(0, 50.0)
        x = self.avgpool(x)

        if x.size(1) == 2 * self.split_size:
            x[:, self.split_size:, :, :] /= float(self.avgpool.kernel_size ** 2)

        encoding = x.view(x.size(0), -1)

        encoding_s = encoding

        if self.training and self.fds:
            encoding_s = self.FDS.smooth(encoding_s, targets, epoch)

        mu, log_var = None, None
        x_recons = None

        if self.use_prm:
            mu, var = torch.tensor_split(encoding_s, 2, dim=1)
            log_var = var.log()

            if self.training:
                sigma = torch.exp(log_var / 2.0)
                encoding_s = mu + torch.empty_like(sigma).normal_(0, 1.0) * sigma
            else:
                encoding_s = mu

        if self.use_dropout:
            encoding_s = self.dropout(encoding_s)


        if self.use_recons:
            encoding_r = encoding_s.view(encoding_s.size(0), encoding_s.size(1), 1, 1)
            encoding_r = self.reverse_avgpool(encoding_r)
            encoding_r = self.reverse_layer4(encoding_r)
            encoding_r = self.reverse_layer3(encoding_r)
            encoding_r = self.reverse_layer2(encoding_r)
            encoding_r = self.reverse_layer1(encoding_r)
            x_recons = self.reverse_initial(encoding_r)

        if self.use_edl:
            if self.use_cdm:
                x = self.conjugate_prior(encoding_s)
            else:
                x = self.linear_normal_edl(encoding_s)
        else:
            x = self.linear(encoding_s)

        if self.training:
            if self.fds:
                if self.use_recons:
                    return x, mu, log_var, encoding, x_recons
                else:
                    return x, mu, log_var, encoding
            else:
                if self.use_recons:
                    return x, None, None, x_recons
                else:
                    return x, None, None
        else:
            return x


def resnet50(**kwargs):
    return ResNet(Bottleneck, ReverseBottleneck, [3, 4, 6, 3], **kwargs)
