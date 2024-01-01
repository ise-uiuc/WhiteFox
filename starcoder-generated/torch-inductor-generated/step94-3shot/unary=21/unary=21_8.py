
import torch
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(128, 64, (3, 3), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=(1, 1), bias=False)
        self.conv2 = torch.nn.Conv2d(1, 1, (3, 3), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=(1, 1), bias=False)
    def forward(self, x):
        v1 = torch.nn.functional.relu(x)
        v2 = torch.tanh(v1.clone())
        return v2.expand_as(x)
# Inputs to the model
x = torch.randn(1, 128, 291, 291)
