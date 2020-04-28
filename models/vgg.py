'''
Author: Sai Aparna Aketi
'''
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from layers import MaskedConv2d 
from layers import MaskedLinear


__all__ = [ 'vgg16_bn']


model_urls = {
  
    'vgg16_bn': 'http://download.pytorch.org/models/vgg16_bn-6c64b313.pth'
}


class VGG1(nn.Module):

    def __init__(self, features, num_classes=10, n =1, init_weights=True):
        super(VGG1, self).__init__()
        self.features = features

        self.classifier = nn.Sequential(
            MaskedLinear(512*n*n, 512),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            MaskedLinear(512, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
                


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = MaskedConv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'vgg_16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
   }




def vgg16_bn(classes=10,n=1, **kwargs):
    model = VGG1(make_layers(cfg['vgg_16'], batch_norm=True), num_classes=classes, n=n,**kwargs)
    return model

