
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = torch.nn.LeakyReLU()
        self.conv = nn.Conv1d(512, 1792, 1)
        conv = torch.nn.Conv1d(512, 1792, 1)
        input_tensor = torch.randn(1, 512, 100)
        self.layer_bn = nn.BatchNorm1d(1792, track_running_stats=True)

        module_out = conv(input_tensor)
        functional_out = F.conv1d(input_tensor, conv.weight, conv.bias, conv.stride[0], conv.padding[0], conv.dilation[0], conv.groups)

        # Should be same
        self.output1 = self.activation(module_out)
        self.output2 = self.activation(functional_out)

        # Should be None
        self.output3 = self.layer_bn(module_out)
        self.output4 = self.layer_bn(functional_out)
    def forward(self, x):
        return self.output3
# Inputs to the model
x = torch.randn(1, 3, 100)
