

import torch
import torch.nn as nn
class m1(nn.Module):
    def __init__(self):
        super().__init__()
        self.m1=nn.ConvTranspose2d(81, 33, 3, stride=2,padding=1,bias=False)
        self.t1=nn.ConvTranspose2d(81, 11, 3, stride=2,padding=1, bias=False)
    def forward(self,x):
        x1=self.m1(x)
        x2 = self.t1(x)
        x3 = x2 > 0
        x4 = x2 * -0.4
        x5 = torch.where(x3, x2, x4)
        x5 =torch.nn.functional.relu(x5)
        x5 =torch.nn.functional.avg_pool2d(x5, (4, 4))
        return torch.nn.functional.tanh(x1)

import torch
import torch.nn as nn
class m1(nn.Module):
    def __init__(self):
        super().__init__()
        self.m1=nn.ConvTranspose2d(81, 33, 3, padding=1,stride=2,bias=False) 
        self.m2=nn.ConvTranspose2d(25, 105, 7,stride=1,padding=0, bias=False)  
    def forward(self,x):
        x1=self.m1(x)
        x2 = self.m2(x)
        x3=x2>0
        x4 = x2 * -0.0245
        x5=torch.where(x3, x2, x4)
        x5 = torch.nn.functional.softmin(x5, dim=-1)
        return torch.nn.functional.linear(x5, 2)


import torch
import torch.nn as nn
class m2(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1=nn.ConvTranspose2d(43, 49, 5, stride=1, padding=3, output_padding=2,bias=False,dilation=2)
        self.c2=nn.ConvTranspose2d(95, 86, 7, stride=3, padding=3, bias=False,groups=4)
        self.c3=nn.ConvTranspose2d(36, 31, 3, stride=1, padding=1, bias=True)
    def forward(self,x):
        x1=self.c1(x)
        x2=self.c2(x)
        x3=self.c3(x)
        x4=x2+x3
        x5=x1-x4
        x6=x4*2
        x7=x5>0
        x8=x4-2
        x9=-0.8*x6
        x11=torch.where(x7, x9, x8)
        return torch.sqrt(x11)
# Inputs to the model
x = torch.randn(1, 43, 2, 2)
