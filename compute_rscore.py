"""
Author: Sai Aparna Aketi
Reference: https://github.com/Hey1Li/Salient-Relevance-Propagation/blob/master/Imagenet_Alex%20vs%20VGG16.ipynb
"""

from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

# defining relevance propagation function for each type of layer
####################################################################################
class Linear(nn.Linear):
    def __init__(self, linear,alpha=2):
        super(nn.Linear, self).__init__()
        self.in_features  = linear.in_features
        self.out_features = linear.out_features
        self.weight       = linear.weight
        self.bias         = linear.bias
        self.alpha        = alpha
        self.beta         = alpha - 1
        
    def relprop(self, R):
        wp = torch.clamp(self.weight, min=0)
        ap = torch.mm(self.X, torch.transpose(wp,0,1)) + 1e-9       
        Sp = (self.alpha * R) / ap
        Cp = torch.mm(Sp, wp)
        wn = torch.clamp(self.weight, max=0)
        an = torch.mm(self.X, torch.transpose(wn,0,1)) + 1e-9
        Sn = (-self.beta * R)/ an
        Cn = torch.mm(Sn, wn)
        R  = self.X * (Cp+Cn)
        return R
       
class ReLU(nn.ReLU):   
    def relprop(self, R): 
        return R

class BatchNorm2d(nn.BatchNorm2d):  
    def __init__(self, batchnorm2d):
        super(nn.BatchNorm2d, self).__init__(batchnorm2d.num_features)
        self.weight       = batchnorm2d.weight
        self.bias         = batchnorm2d.bias
        self.running_mean = batchnorm2d.running_mean
        self.running_var  = batchnorm2d.running_var        
        
    def relprop(self, R): 
        return R

class Reshape(nn.Module):
    def __init__(self,f,n):
        super(Reshape, self).__init__()
        self.f = f
        self.n = n
        
    def forward(self, x):
        return x.view(-1, self.f*self.n*self.n)
        
    def relprop(self, R):
        return R.view(-1, self.f,self.n,self.n)
    
class ResConnect_add(nn.Module):
    def __init__(self):
        super(ResConnect_add, self).__init__()
        
    def forward(self, x):
        return x
        
    def relprop(self, R1,R2):
        return R1+R2

    
class ResConnect_split(nn.Module):
    def __init__(self):
        super(ResConnect_split, self).__init__()
        
    def forward(self, x, y):
        return x+y
        
    def relprop(self, R):
        x1 = self.X
        x2 = self.Y - self.X
        Z = self.Y + 1e-9
        S = R / Z
        R = x1 * S
        Rprev = x2 * S
        return R , Rprev

class MaxPool2d(nn.MaxPool2d):
    def __init__(self, maxpool2d, alpha = 2):
        super(nn.MaxPool2d, self).__init__(maxpool2d.kernel_size)
        self.kernel_size    = maxpool2d.kernel_size
        self.stride         = maxpool2d.stride
        self.padding        = maxpool2d.padding
        self.dilation       = maxpool2d.dilation
        self.return_indices = maxpool2d.return_indices
        self.ceil_mode      = maxpool2d.ceil_mode
        self.alpha          = alpha
        self.beta           = alpha-1
        
    def gradprop(self, DY):
        DX = self.X * 0
        temp, indices = F.max_pool2d(self.X, self.kernel_size, self.stride, 
                                     self.padding, self.dilation, self.ceil_mode, True)
        DX = F.max_unpool2d(DY, indices, self.kernel_size, self.stride, self.padding)
        return DX
    
    def relprop(self, R):
        Z = self.Y + 1e-9
        S = R / Z
        C = self.gradprop(S)
        R = self.X * C
        return R


class AvgPool2d(nn.AvgPool2d):
    def __init__(self, avgpool2d, alpha = 2):
        super(nn.AvgPool2d, self).__init__()
        self.kernel_size        = avgpool2d.kernel_size
        self.stride             = avgpool2d.stride
        self.padding            = avgpool2d.padding
        self.count_include_pad  = avgpool2d.count_include_pad
        self.ceil_mode          = avgpool2d.ceil_mode
        self.divisor_override   = avgpool2d.divisor_override
        self.alpha          = alpha
        self.beta           = alpha-1
        
        
    def gradprop(self, DY):
        
        DX = F.interpolate(DY, scale_factor = self.kernel_size)
        return DX
    
    def relprop(self, R):
        Z = self.Y + 1e-9
        S = R / Z
        C = self.gradprop(S)
        R = self.X * C
        return R
    

    
class Conv2d(nn.Conv2d):
    def __init__(self, conv2d, alpha=2):
        super(nn.Conv2d, self).__init__(conv2d.in_channels, 
                                        conv2d.out_channels, 
                                        conv2d.kernel_size, 
                                        conv2d.stride, 
                                        conv2d.padding, 
                                        conv2d.dilation, 
                                        conv2d.transposed, 
                                        conv2d.output_padding, 
                                        conv2d.groups, 
                                        True, padding_mode = 'zeros')
        self.weight = conv2d.weight
        self.bias   = conv2d.bias
        self.alpha  = alpha
        self.beta   = alpha-1
       
    def forprop(self, x,V):
        return F.conv2d(x, V, stride=self.stride, padding=self.padding)
        
        
    def gradprop(self, DY,V):
        output_padding = self.X.size()[2] - ((self.Y.size()[2] - 1) * self.stride[0] \
                                             - 2 * self.padding[0] + self.kernel_size[0])
        return F.conv_transpose2d(DY, V, stride=self.stride, 
                                  padding=self.padding, output_padding=output_padding)
        
    def relprop(self, R):
        Vp = torch.clamp(self.weight, min=1e-9)
        Vn = torch.clamp(self.weight, max= (0-1e-9))
        X  = self.X 
        Zp = self.forprop(X,Vp)+ 1e-9
        Zn = self.forprop(X,Vn)+ 1e-9
        Sp =(self.alpha* R) / Zp
        Sn =(-self.beta* R) / Zn
        Cp = self.gradprop(Sp,Vp)
        Cn = self.gradprop(Sn,Vn)
        R  = self.X *(Cp + Cn)
        return R
########################################
