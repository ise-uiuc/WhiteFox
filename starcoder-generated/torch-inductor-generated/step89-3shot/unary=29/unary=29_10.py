
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils

class Model(nn.Module):

    def __init__(self, min_value=-1.3402, max_value=1.8860):
        super(Model, self).__init__()
        self.conv_bn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=1, stride=1, padding=1), # Input shape: [1,3, 64,64]
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1), # Input shape: [1,16, 32,32]
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 1, kernel_size=8, stride=2, padding=1), # Input shape: [1,32, 16,16]
            nn.BatchNorm2d(1)
        )
        self.min_value = min_value
        self.max_value = max_value
        
    def forward(self, x1):
        batch_size = x1.shape[0]
        x1 = x1.view(batch_size, 3, 64, 64)
        v1 = self.conv_bn(x1)
        v1 = torch.permute(v1, (0, 2, 3, 1)) # Shape: [1, 64,64, 32]
        v2 = torch.clamp_min(v1, self.min_value)
        v2 = torch.clamp_max(v2, self.max_value)
        v3 = torch.permute(v2, (0, 3, 1, 2)) # Shape: [1, 32,64, 64] 
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
