
import torch.nn as nn
import numpy as np
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(8, 16, 7, stride=(2,2,1), padding=(1,1,0))
        self.conv2 = nn.Conv3d(8, 16, 7, stride=(2,2,1), padding=(1,1,0))
        self.conv3 = nn.Conv3d(8, 16, 3, stride=(2,1,1), padding=(1,0,0))
        self.conv4 = nn.Conv3d(8, 16, 3, stride=(2,1,1), padding=(1,0,0))
        self.pool = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        #self.maxpool = nn.MaxPool3d(kernel_size=(2, 2, 1), stride=(1, 1, 1), padding=(1, 1, 0))
        #self.avgpool = nn.AvgPool3d(kernel_size=(2, 2, 1), stride=(1, 1, 1), padding=(1, 1, 0))
        self.bn1 = nn.BatchNorm3d(16, eps=False)
        self.conv5 = nn.Conv3d(16, 32, 7, stride=(2,2,1), padding=(1,1,0))
        self.bn2 = nn.BatchNorm3d(32, eps=False)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(x)
        v3 = self.conv3(x)
        v4 = self.conv4(x)
        v5 = self.bn1(v1) + v2 + v3 + v4
        v6 = self.pool(v5)
        v7 = v6.squeeze()
        v8 = self.bn2(self.conv5(v7))
        v9 = self.relu(v8)
        return v9

# Inputs to the model
x = torch.randn(1, 8, 64, 224, 224)
