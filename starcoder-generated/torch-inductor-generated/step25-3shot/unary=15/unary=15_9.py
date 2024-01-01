
import torch
import torch.nn as nn

class Mish(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            x = x * (torch.tanh(nn.functional.softplus(x)))
            return x

class Model(nn.Module):
    def __init__(self,in_channel,out_channel,kernelSize=3):
        super(Model,self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernelSize)
        self.pool = nn.MaxPool2d(kernelSize=2)
        # self.activation = nn.PReLU()
        self.dropout = nn.Dropout2d(p=0.25)
        # self.activation = nn.ReLU(inplace=True)
        self.mish = Mish()
    # self.activation = nn.Softmax(dim=1)
    # input N,64,64
    def forward(self,x):
        x = self.conv(x)
        x = self.mish(x)
        x = self.pool(x)
        x = self.dropout(x)
        # x = self.mish(x)
        # x = self.pool(x)
        # x = self.activation(x)
        return x
# Inputs to the model
in_channel = 3
out_channel = 8
x1 = torch.randn(1, in_channel, 64, 64)
