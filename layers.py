import torch
import torch.nn as nn
import torch.nn.functional as F
from utils_1 import to_var

        
        
class MaskedConv2d(nn.Conv2d):
	def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False):
		super(MaskedConv2d, self).__init__(in_channels, out_channels, 
		    kernel_size, stride, padding, dilation, groups, bias)
		self.mask_flag = False
    
	def set_mask(self, mask):
		self.mask = to_var(mask, requires_grad=False); 
		self.weight.data = self.weight.data*self.mask.data; 
		self.mask_flag = True
        
	def get_mask(self):
		print(self.mask_flag)
		return self.mask
    
	def forward(self, x):
		if self.mask_flag == True:
			weight = self.weight.cuda()*self.mask.cuda()
			self.weight.data = self.weight.data.cuda()*self.mask.data.cuda()
			return F.conv2d(x, weight, self.bias, self.stride,self.padding, self.dilation, self.groups)
		else:
            		
			return F.conv2d(x, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class MaskedLinear(nn.Linear):
	def __init__(self, in_features, out_features, bias=True):
		super(MaskedLinear, self).__init__(in_features, out_features, bias)
		self.mask_flag = False
    
	def set_mask(self, mask, mask_b):
		self.mask = to_var(mask, requires_grad=False); self.mask_b = to_var(mask_b, requires_grad=False)
		self.weight.data = self.weight.data*self.mask.data; self.bias.data = self.bias.data*self.mask_b.data
		self.mask_flag = True

	def get_mask(self):
		print(self.mask_flag)
		return self.mask
    
	def forward(self, x):
		if self.mask_flag == True:
			weight = self.weight.cuda()*self.mask.cuda(); 
			self.weight.data = self.weight.data.cuda()*self.mask.data.cuda(); 
			bias = self.bias.cuda()*self.mask_b.cuda()
			return F.linear(x, weight, bias)
		else:
			return F.linear(x, self.weight, self.bias)
        
