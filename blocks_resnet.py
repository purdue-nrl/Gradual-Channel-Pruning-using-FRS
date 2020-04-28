'''
Author: Sai Aparna Aketi
'''
from __future__ import print_function
import torch.nn as nn
from compute_rscore import *
import torch.nn.functional as F

class block(nn.Module):
    def __init__(self, layer):
        super(block, self).__init__()
        self.lay = nn.Sequential(
            Conv2d(layer.conv1),       
            BatchNorm2d(layer.bn1),
            ReLU(),
            Conv2d(layer.conv2),       
            BatchNorm2d(layer.bn2)) 
        self.con = ResConnect_split()
        self.res = nn.Sequential(
                ReLU(), 
                ResConnect_add())
        
    def forward(self, x):
        out = self.lay(x)
        out = self.con(out, x)
        out = self.res(out)
        return out
        
    def relprop(self, R, Rprev):
        # propagates the relevance score through the block module  
        R = self.res[1].relprop(R, Rprev)    
        R = self.res[0].relprop(R)
        self.res[0].register_buffer('Rscore',R)
        R, Rprev = self.con.relprop(R) 
        for l in range(len(self.lay), 0, -1):
            R = self.lay[l-1].relprop(R)
            self.lay[l-1].register_buffer('Rscore',R)  
        return R, Rprev
    
##############################################################################
        
class block_with_shortcut(nn.Module):
    def __init__(self, layer):
        super(block_with_shortcut, self).__init__()
        self.lay = nn.Sequential(
            Conv2d(layer.conv1),       
            BatchNorm2d(layer.bn1),
            ReLU(),
            Conv2d(layer.conv2),       
            BatchNorm2d(layer.bn2)) 
        self.con = ResConnect_split()
        self.res = nn.Sequential(
                ReLU(), 
                ResConnect_add())
        self.shortcut = nn.Sequential(
            Conv2d(layer.shortcut[0]),       
            BatchNorm2d(layer.shortcut[1]))
        
    def forward(self, x):
        out = self.lay(x)
        out = self.con(out, self.shortcut(x))
        out = self.res(out)
        return out
        
    def relprop(self, R, Rprev):
        # propagates the relevance score through the block module  
        R = self.res[1].relprop(R, Rprev)    
        R = self.res[0].relprop(R)
        self.res[0].register_buffer('Rscore',R)
        R, Rprev = self.con.relprop(R)
        for l in range(len(self.shortcut), 0, -1):
            Rprev = self.shortcut[l-1].relprop(Rprev)
            self.shortcut[l-1].register_buffer('Rscore',Rprev)            
        for l in range(len(self.lay), 0, -1):
            R = self.lay[l-1].relprop(R)
            self.lay[l-1].register_buffer('Rscore',R) 
        return R, Rprev

##############################################################################
        
class block_end(nn.Module):
    def __init__(self, layer):
        super(block_end, self).__init__()
        self.lay = nn.Sequential(
            Conv2d(layer.conv1),       
            BatchNorm2d(layer.bn1),
            ReLU(),
            Conv2d(layer.conv2),       
            BatchNorm2d(layer.bn2)) 
        self.con = ResConnect_split()
        self.res = nn.Sequential(
                ReLU())
        
    def forward(self, x):
        out = self.lay(x)
        out = self.con(out, x)
        out = self.res(out)
        return out
        
    def relprop(self, R):
        # propagates the relevance score through the block_end module    
        R = self.res[0].relprop(R)
        self.res[0].register_buffer('Rscore',R)
        R, Rprev = self.con.relprop(R) 
        for l in range(len(self.lay), 0, -1):
            R = self.lay[l-1].relprop(R)
            self.lay[l-1].register_buffer('Rscore',R)  
        return R, Rprev
    
##############################################################################
        
class bottleneck_block(nn.Module):
    def __init__(self, layer):
        super(bottleneck_block, self).__init__()
        self.lay = nn.Sequential(
            Conv2d(layer.conv1),       
            BatchNorm2d(layer.bn1),
            ReLU(),
            Conv2d(layer.conv2),       
            BatchNorm2d(layer.bn2),
            ReLU(),
            Conv2d(layer.conv3),       
            BatchNorm2d(layer.bn3)) 
        self.con = ResConnect_split()
        self.res = nn.Sequential(
                ReLU(), 
                ResConnect_add())
        
    def forward(self, x):
        out = self.lay(x)
        out = self.con(out, x)
        out = self.res(out)
        return out
        
    def relprop(self, R, Rprev):
        R = self.res[1].relprop(R, Rprev)    
        R = self.res[0].relprop(R)
        self.res[0].register_buffer('Rscore',R)
        R, Rprev = self.con.relprop(R) 
        for l in range(len(self.lay), 0, -1):
            R = self.lay[l-1].relprop(R)
            self.lay[l-1].register_buffer('Rscore',R)  
        return R, Rprev
##############################################################################
class bottleneck_shortcut(nn.Module):
    def __init__(self, layer):
        super(bottleneck_shortcut, self).__init__()
        self.lay = nn.Sequential(
            Conv2d(layer.conv1),       
            BatchNorm2d(layer.bn1),
            ReLU(),
            Conv2d(layer.conv2),       
            BatchNorm2d(layer.bn2),
            ReLU(),
            Conv2d(layer.conv3),       
            BatchNorm2d(layer.bn3)) 
        self.con = ResConnect_split()
        self.res = nn.Sequential(
                ReLU(), 
                ResConnect_add())
        self.shortcut = nn.Sequential(
            Conv2d(layer.shortcut[0]),       
            BatchNorm2d(layer.shortcut[1]))
        
    def forward(self, x):
        out = self.lay(x)
        out = self.con(out, self.shortcut(x))
        out = self.res(out)
        return out
        
    def relprop(self, R, Rprev):
        # propagates the relevance score through the block module  
        R = self.res[1].relprop(R, Rprev)    
        R = self.res[0].relprop(R)
        self.res[0].register_buffer('Rscore',R)
        R, Rprev = self.con.relprop(R)
        for l in range(len(self.shortcut), 0, -1):
            Rprev = self.shortcut[l-1].relprop(Rprev)
            self.shortcut[l-1].register_buffer('Rscore',Rprev)            
        for l in range(len(self.lay), 0, -1):
            R = self.lay[l-1].relprop(R)
            self.lay[l-1].register_buffer('Rscore',R) 
        return R, Rprev

##############################################################################
        
class bottleneck_end(nn.Module):
    def __init__(self, layer):
        super(bottleneck_end, self).__init__()
        self.lay = nn.Sequential(
            Conv2d(layer.conv1),       
            BatchNorm2d(layer.bn1),
            ReLU(),
            Conv2d(layer.conv2),       
            BatchNorm2d(layer.bn2),
            ReLU(),
            Conv2d(layer.conv3),       
            BatchNorm2d(layer.bn3)) 
        self.con = ResConnect_split()
        self.res = nn.Sequential(
                ReLU())
        
    def forward(self, x):
        out = self.lay(x)
        out = self.con(out, x)
        out = self.res(out)
        return out
        
    def relprop(self, R):
        # propagates the relevance score through the block_end module    
        R = self.res[0].relprop(R)
        self.res[0].register_buffer('Rscore',R)
        R, Rprev = self.con.relprop(R) 
        for l in range(len(self.lay), 0, -1):
            R = self.lay[l-1].relprop(R)
            self.lay[l-1].register_buffer('Rscore',R)  
        return R, Rprev
    
