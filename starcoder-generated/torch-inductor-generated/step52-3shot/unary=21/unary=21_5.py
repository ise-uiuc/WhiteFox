
import torch.nn as nn
class ModelTanh(nn.Module):
    def __init__(self):
        super(ModelTanh, self).__init__()
        self.conv_1 = nn.Conv2d(1, 4, 1)
    def forward(self, x1):
        x2 = self.conv_1(x1)
        x2 = x2[:, :, 1:-1, 1:-1]
        x3 = torch.tanh(x2)
        return x3
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
