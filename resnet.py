# This resnet implementation is modified based on torchvision.models.resnet
import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import pdb


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


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
        self.has_lens = False
        
    def forward(self, x):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        lens_out = {}
        if self.has_lens:
            # out: 512 channels, residual: 2048 channels
            lens_out['pre_conv'] = out.clone()
            lens_out['shortcut'] = residual.clone()

        out = self.conv2(out)
        out = self.bn2(out)
        out += residual

        if self.has_lens:
            lens_out['pre_relu'] = out.clone()
                
        out = self.relu(out)
        if self.has_lens:
            lens_out['out'] = out
            return lens_out
        else:
            return out
            

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride
        self.has_lens = False
        
    def forward(self, x):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        lens_out = {}
        if self.has_lens:
            # out: 512 channels, residual: 2048 channels
            lens_out['pre_conv'] = out.clone()
            lens_out['shortcut'] = residual.clone()
            
        out = self.conv3(out)
        out = self.bn3(out)
        out += residual
        
        if self.has_lens:
            lens_out['pre_relu'] = out.clone()
                
        out = self.relu(out)
        if self.has_lens:
            lens_out['out'] = out
            return lens_out
        else:
            return out

# set do_pool1=False when taking small images as input
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, lens_stage=None,
                 no_fc=False, no_avgpool=False, do_pool1=True):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.create_xlayer = False
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)

        self._lens_stage = lens_stage
        if 'xlayer' == self._lens_stage:
            self.create_xlayer = True
            
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.no_fc = no_fc
        self.no_avgpool = no_avgpool
        self.do_pool1 = do_pool1
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # for simplicity, assume lenses are inserted after one particular stage only 
        if 'res3' == self._lens_stage:
            self.layer3[-1].has_lens = True
        if 'res4' == self._lens_stage:
            self.layer4[-1].has_lens = True
        self.pass_featlens = False
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        
        if self.create_xlayer:
            blocks += 1
            print("extra layer created in res4")   
                     
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def ext_features(self, x):
        x0 = self.conv1(x)
        x0_bn = self.bn1(x0)
        x0_relu = self.relu(x0_bn)
        if self.do_pool1:
            x0_pool = self.maxpool(x0_relu)
        else:
            x0_pool = x0_relu

        x1 = self.layer1(x0_pool)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        
        return x0_pool, x1, x2, x3, x4

    def set_pass_featlens(self, pass_featlens):
        self.pass_featlens = pass_featlens
    
    def filter_feats(self, x, stage_name):
        if type(x) == dict:
            # during training or test in train.py, always pass_featlens = True
            if self.pass_featlens and 'lenskit' in self._modules \
              and stage_name == self.lenskit.host_stage_name:
                x = self.lenskit(x)
            else:
                x = x['out']
                
        return x
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.do_pool1:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        
        x = self.layer3(x)
        x = self.filter_feats(x, 'res3')
        
        if 'xlayer' == self._lens_stage:
            # Make sure xlayer and downstream always get gradients
            with torch.enable_grad():
                x = self.layer4(x)
                # x = self.filter_feats(x, 'res4')
                x = self.avgpool(x)
                x = x.view(x.size(0), -1)
                
                if self.no_fc:
                    return x
                    
                x = self.fc(x)
                self.cls_scores = x
                return x

        x = self.layer4(x)
        x = self.filter_feats(x, 'res4')
                    
        if self.no_avgpool:
            return x
            
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        if self.no_fc:
            return x
            
        x = self.fc(x)

        # if not xlayer, no need to assign self.cls_scores = x
        # as the returned x has gradients
        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']), strict=False)
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']), strict=False)
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']), strict=False)
    return model
