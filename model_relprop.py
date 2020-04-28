'''
Author: Sai Aparna Aketi
'''
from __future__ import print_function
import torch.nn as nn
from compute_rscore import *
import torch.nn.functional as F
from blocks_resnet import *

###############################################################
class VGG16Net_cifar(nn.Module):
    def __init__(self,vgg16):
        super(VGG16Net_cifar, self).__init__()
        self.layers = nn.Sequential(
            Conv2d(vgg16.module.features[0]),       #1
            BatchNorm2d(vgg16.module.features[1]),
            ReLU(),
            Conv2d(vgg16.module.features[3]),       #2
            BatchNorm2d(vgg16.module.features[4]),
            ReLU(),
            MaxPool2d(vgg16.module.features[6]),
            Conv2d(vgg16.module.features[7]),       #3
            BatchNorm2d(vgg16.module.features[8]),
            ReLU(),
            Conv2d(vgg16.module.features[10]),      #4
            BatchNorm2d(vgg16.module.features[11]),
            ReLU(),
            MaxPool2d(vgg16.module.features[13]),
            Conv2d(vgg16.module.features[14]),      #5
            BatchNorm2d(vgg16.module.features[15]),
            ReLU(),
            Conv2d(vgg16.module.features[17]),      #6
            BatchNorm2d(vgg16.module.features[18]),
            ReLU(),
            Conv2d(vgg16.module.features[20]),      #7
            BatchNorm2d(vgg16.module.features[21]),
            ReLU(),
            MaxPool2d(vgg16.module.features[23]),
            Conv2d(vgg16.module.features[24]),      #8
            BatchNorm2d(vgg16.module.features[25]),
            ReLU(),
            Conv2d(vgg16.module.features[27]),      #9
            BatchNorm2d(vgg16.module.features[28]),
            ReLU(),
            Conv2d(vgg16.module.features[30]),      #10
            BatchNorm2d(vgg16.module.features[31]),
            ReLU(),
            MaxPool2d(vgg16.module.features[33]),
            Conv2d(vgg16.module.features[34]),      #11
            BatchNorm2d(vgg16.module.features[35]),
            ReLU(),
            Conv2d(vgg16.module.features[37]),      #12
            BatchNorm2d(vgg16.module.features[38]),
            ReLU(),
            Conv2d(vgg16.module.features[40]),      #13
            BatchNorm2d(vgg16.module.features[41]),
            ReLU(),
            MaxPool2d(vgg16.module.features[43]),
            Reshape(f=512, n=1),
            Linear(vgg16.module.classifier[0]),
            ReLU(),
            Linear(vgg16.module.classifier[3])
            )
              
    def forward(self, x):
        out  = self.layers(x)
        return out
    
    def relprop(self, R):
        for l in range(len(self.layers), 0, -1):
            R = self.layers[l-1].relprop(R)
            self.layers[l-1].register_buffer('Rscore',R)
        return R

    
##############################################################################
    
class RES56Net(nn.Module):
    def __init__(self,res56):
        super(RES56Net, self).__init__()
        
        self.initial    = nn.Sequential(Conv2d(res56.module.conv1),BatchNorm2d(res56.module.bn1),ReLU())
        self.res_i      = nn.Sequential(ResConnect_add())
        self.layer1     = self._make_layer(block,res56.module.layer1, 9)
        self.layer2_s   = block_with_shortcut(res56.module.layer2[0])
        self.layer2     = self._make_layer_s(block,res56.module.layer2, 9)  
        self.layer3_s   = block_with_shortcut(res56.module.layer3[0])
        self.layer3     = self._make_layer_s(block,res56.module.layer3, 8) 
        self.layer3_end = block_end(res56.module.layer3[8])  
        self.avgpool    = AvgPool2d(res56.module.avgpool) 
        self.reshape    = Reshape(f=64, n=1)
        self.linear     = Linear(res56.module.fc)
        self.softmax    = nn.Softmax(dim=1)
    
    def _make_layer(self, block, layer, count):
        layers = []
        layers.append(block(layer[0]))
        for i in range(1, count):
            layers.append(block(layer[i]))
        return nn.Sequential(*layers)
        
    def _make_layer_s(self, block, layer, count):
        layers = []
        layers.append(block(layer[1]))
        for i in range(2, count):
            layers.append(block(layer[i]))
        return nn.Sequential(*layers)   
        
        
    def forward(self, x): 
        x   = self.initial(x)
        x   = self.res_i(x)
        out = self.layer1(x)            
        out = self.layer2_s(out)
        out = self.layer2(out)            
        out = self.layer3_s(out)
        out = self.layer3(out)
        out = self.layer3_end(out)
        out = self.avgpool(out)
        out = self.linear(self.reshape(out))
        out = self.softmax(out)
        return out
    
    def relprop(self, R):
        R = self.linear.relprop(R)
        R = self.reshape.relprop(R)
        R = self.avgpool.relprop(R)
        R, Rprev = self.layer3_end.relprop(R) 
        for i in range(0,7):              
            R, Rprev = self.layer3[6-i].relprop(R, Rprev)                   
        R, Rprev = self.layer3_s.relprop(R, Rprev)
        for i in range(0,8):    
            R, Rprev = self.layer2[7-i].relprop(R, Rprev)                   
        R, Rprev = self.layer2_s.relprop(R, Rprev)
        for i in range(0,9):
            R, Rprev = self.layer1[8-i].relprop(R, Rprev) 
        return R
################################################################################
        
class RES110Net(nn.Module):
    def __init__(self,res110, n=1):
        super(RES110Net, self).__init__()
        
        self.initial    = nn.Sequential(Conv2d(res110.module.conv1),BatchNorm2d(res110.module.bn1),ReLU())
        self.res_i      = nn.Sequential(ResConnect_add())
        self.layer1     = self._make_layer(block,res110.module.layer1, 18)
        self.layer2_s   = block_with_shortcut(res110.module.layer2[0])
        self.layer2     = self._make_layer_s(block,res110.module.layer2, 18)  
        self.layer3_s   = block_with_shortcut(res110.module.layer3[0])
        self.layer3     = self._make_layer_s(block,res110.module.layer3, 17) 
        self.layer3_end = block_end(res110.module.layer3[17]) 
        self.avgpool    = AvgPool2d(res110.module.avgpool) 
        self.reshape    = Reshape(f=64,n=n)
        self.linear     = Linear(res110.module.fc)
        self.softmax    = nn.Softmax(dim=1)
    
    def _make_layer(self, block, layer, count):
        layers = []
        layers.append(block(layer[0]))
        for i in range(1, count):
            layers.append(block(layer[i]))
        return nn.Sequential(*layers)
        
    def _make_layer_s(self, block, layer, count):
        layers = []
        layers.append(block(layer[1]))
        for i in range(2, count):
            layers.append(block(layer[i]))
        return nn.Sequential(*layers)   
        
        
    def forward(self, x): 
        x    = self.initial(x)
        x    = self.res_i(x)    
        out  = self.layer1(x)          
        out  = self.layer2_s(out)
        out  = self.layer2(out)            
        out  = self.layer3_s(out)
        out  = self.layer3(out)
        out  = self.layer3_end(out)
        out  = self.avgpool(out)
        out  = self.linear(self.reshape(out))
        out  = self.softmax(out)
        return out
    
    def relprop(self, R):
        R = self.linear.relprop(R)
        R = self.reshape.relprop(R)
        R = self.avgpool.relprop(R)
        R, Rprev = self.layer3_end.relprop(R) 
        for i in range(0,16):              
            R, Rprev = self.layer3[15-i].relprop(R, Rprev)                   
        R, Rprev = self.layer3_s.relprop(R, Rprev)
        for i in range(0,17):    
            R, Rprev = self.layer2[16-i].relprop(R, Rprev)                   
        R, Rprev = self.layer2_s.relprop(R, Rprev)
        for i in range(0,18):
            R, Rprev = self.layer1[17-i].relprop(R, Rprev) 
        return R

#####################################################################
        
class RES164Net(nn.Module):
    def __init__(self,res164, n=1):
        super(RES164Net, self).__init__()
        
        self.initial    = nn.Sequential(Conv2d(res164.module.conv1),BatchNorm2d(res164.module.bn1),ReLU())
        self.res_i      = nn.Sequential(ResConnect_add())
        self.layer1_s   = bottleneck_shortcut(res164.module.layer1[0])
        self.layer1     = self._make_layer_s(bottleneck_block,res164.module.layer1, 18) 
        self.layer2_s   = bottleneck_shortcut(res164.module.layer2[0])
        self.layer2     = self._make_layer_s(bottleneck_block,res164.module.layer2, 18)  
        self.layer3_s   = bottleneck_shortcut(res164.module.layer3[0])
        self.layer3     = self._make_layer_s(bottleneck_block,res164.module.layer3, 17) 
        self.layer3_end = bottleneck_end(res164.module.layer3[17]) 
        self.avgpool    = AvgPool2d(res164.module.avgpool) 
        self.reshape    = Reshape(f=256, n=n)
        self.linear     = Linear(res164.module.fc)
        self.softmax    = nn.Softmax(dim=1)
    
    def _make_layer(self, bottleneck_block, layer, count):
        layers = []
        layers.append(bottleneck_block(layer[0]))
        for i in range(1, count):
            layers.append(bottleneck_block(layer[i]))
        return nn.Sequential(*layers)
        
    def _make_layer_s(self, bottleneck_block, layer, count):
        layers = []
        layers.append(bottleneck_block(layer[1]))
        for i in range(2, count):
            layers.append(bottleneck_block(layer[i]))
        return nn.Sequential(*layers)   
        
        
    def forward(self, x): 
        x    = self.initial(x)
        x    = self.res_i(x)
        out  = self.layer1_s(x)
        out  = self.layer1(out)          
        out  = self.layer2_s(out)
        out  = self.layer2(out)            
        out  = self.layer3_s(out)
        out  = self.layer3(out)
        out  = self.layer3_end(out)
        out  = self.avgpool(out)
        out  = self.linear(self.reshape(out))
        out  = self.softmax(out)
        return out
    
    def relprop(self, R):
        R = self.linear.relprop(R)
        R = self.reshape.relprop(R)
        R = self.avgpool.relprop(R)
        R, Rprev = self.layer3_end.relprop(R)
        for i in range(0,16):              
            R, Rprev = self.layer3[15-i].relprop(R, Rprev)                   
        R, Rprev = self.layer3_s.relprop(R, Rprev)
        for i in range(0,17):    
            R, Rprev = self.layer2[16-i].relprop(R, Rprev)                   
        R, Rprev = self.layer2_s.relprop(R, Rprev)
        for i in range(0,17):    
            R, Rprev = self.layer1[16-i].relprop(R, Rprev)                   
        R, Rprev = self.layer1_s.relprop(R, Rprev)
        return R
    
#################################################################################
        
class RES34Net(nn.Module):
    def __init__(self,res34):
        super(RES34Net, self).__init__()
        
        self.initial    = nn.Sequential(Conv2d(res34.module.conv1),BatchNorm2d(res34.module.bn1),
                                        ReLU(),MaxPool2d(res34.module.maxpool1))
        self.res_i      = nn.Sequential(ResConnect_add())
        self.layer1     = self._make_layer(block,res34.module.layer1, 3)
        self.layer2_s   = block_with_shortcut(res34.module.layer2[0])
        self.layer2     = self._make_layer_s(block,res34.module.layer2, 4)  
        self.layer3_s   = block_with_shortcut(res34.module.layer3[0])
        self.layer3     = self._make_layer_s(block,res34.module.layer3, 6) 
        self.layer4_s   = block_with_shortcut(res34.module.layer4[0])
        self.layer4     = self._make_layer_s(block,res34.module.layer4, 2) 
        self.layer4_end = block_end(res34.module.layer4[2]) 
        self.avgpool    = AvgPool2d(res34.module.avgpool) 
        self.reshape    = Reshape(f=512, n=1)
        self.linear     = Linear(res34.module.fc)
        self.softmax    = nn.Softmax(dim=1)
    
    def _make_layer(self, block, layer, count):
        layers = []
        layers.append(block(layer[0]))
        for i in range(1, count):
            layers.append(block(layer[i]))
        return nn.Sequential(*layers)
        
    def _make_layer_s(self, block, layer, count):
        layers = []
        layers.append(block(layer[1]))
        for i in range(2, count):
            layers.append(block(layer[i]))
        return nn.Sequential(*layers)   
        
        
    def forward(self, x): 
        x   = self.initial(x)
        x   = self.res_i(x)
        out = self.layer1(x)            
        out = self.layer2_s(out)
        out = self.layer2(out)            
        out = self.layer3_s(out)
        out = self.layer3(out)
        out = self.layer4_s(out)
        out = self.layer4(out)
        out = self.layer4_end(out)
        out = self.avgpool(out)
        out = self.linear(self.reshape(out))
        out = self.softmax(out)
        return out
    
    def relprop(self, R):
        R = self.linear.relprop(R)
        R = self.reshape.relprop(R)
        R = self.avgpool.relprop(R)
        R, Rprev = self.layer4_end.relprop(R) 
        R, Rprev = self.layer4[0].relprop(R, Rprev)                   
        R, Rprev = self.layer4_s.relprop(R, Rprev)
        for i in range(0,5):              
            R, Rprev = self.layer3[4-i].relprop(R, Rprev)                   
        R, Rprev = self.layer3_s.relprop(R, Rprev)
        for i in range(0,3):    
            R, Rprev = self.layer2[2-i].relprop(R, Rprev)                   
        R, Rprev = self.layer2_s.relprop(R, Rprev)
        for i in range(0,3):
            R, Rprev = self.layer1[2-i].relprop(R, Rprev) 
        return R
