import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from utils import DownsampleA, DownsampleC, DownsampleD
import math
from torch_deform_conv.layers import ConvOffset2D
from resnet import BasicBlock as ResnetBasicBlock
from resnet import Bottleneck as ResnetBottleneck
from resnet import conv3x3

class DeformResNetBasicblock(nn.Module):
  expansion = 1

  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(DeformResNetBasicblock, self).__init__()

    self.conv_1 = conv3x3(inplanes, planes, stride)
    self.bn_1 = nn.BatchNorm2d(planes)

    self.offset_2 = ConvOffset2D(planes)
    self.conv_2 = conv3x3(planes, planes)
    self.bn_2 = nn.BatchNorm2d(planes)

    self.downsample = downsample

  def forward(self, x):
    residual = x

    basicblock = self.conv_1(x)
    basicblock = self.bn_1(basicblock)
    basicblock = F.relu(basicblock, inplace=True)

    basicblock = self.offset_2(basicblock)
    basicblock = self.conv_2(basicblock)
    basicblock = self.bn_2(basicblock)

    if self.downsample is not None:
      residual = self.downsample(x)

    return F.relu(residual + basicblock, inplace=True)



class DeformResNetBottleneck(nn.Module):
  expansion = 4

  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(DeformResNetBottleneck, self).__init__()

    self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False) # change
    self.bn1 = nn.BatchNorm2d(planes)

    self.offset_2 = ConvOffset2D(planes)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, # change
                           padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(planes)

    self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
    self.bn3 = nn.BatchNorm2d(planes * 4)
    self.relu = nn.ReLU(inplace=True)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):

    residual = x
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.offset_2(out)
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

class CifarDeformResNet(nn.Module):
  """
  ResNet optimized for the Cifar dataset, as specified in
  https://arxiv.org/abs/1512.03385.pdf
  """
  def __init__(self, block_a, block_b, depth, num_classes):
    """ Constructor
    Args:
      depth: number of layers.
      num_classes: number of classes
      base_width: base width
    """
    super(CifarDeformResNet, self).__init__()

    #Model type specifies number of layers for CIFAR-10 and CIFAR-100 model
    assert (depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110'
    layer_blocks = (depth - 2) // 6
    print ('CifarDeformResNet : Depth : {} , Layers for each block : {}'.format(depth, layer_blocks))

    self.num_classes = num_classes

    self.conv_1_3x3 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn_1 = nn.BatchNorm2d(16)

    self.inplanes = 16
    self.stage_1 = self._make_layer(block_a, 16, layer_blocks, 1)
    self.stage_2 = self._make_layer(block_b, 32, layer_blocks, 2)
    self.stage_3 = self._make_layer(block_b, 64, layer_blocks, 2)
    self.avgpool = nn.AvgPool2d(8)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        #m.bias.data.zero_()
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
      elif isinstance(m, nn.Linear):
        init.kaiming_normal(m.weight)
        m.bias.data.zero_()

  def _make_layer(self, block, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = DownsampleC(self.inplanes, planes * block.expansion, stride)

    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(block(self.inplanes, planes))

    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv_1_3x3(x)
    x = F.relu(self.bn_1(x), inplace=True)
    x = self.stage_1(x)
    x = self.stage_2(x)
    x = self.stage_3(x)
    x = self.avgpool(x)
    x = torch.squeeze(x)
    x = F.log_softmax(x)

    return x

def deform_resnet56(num_classes):
  return CifarDeformResNet(ResnetBottleneck, DeformResNetBottleneck, 56, num_classes)

def deform_resnet110(num_classes):
  return CifarDeformResNet(ResnetBasicBlock, DeformResNetBasicblock, 110, num_classes)
