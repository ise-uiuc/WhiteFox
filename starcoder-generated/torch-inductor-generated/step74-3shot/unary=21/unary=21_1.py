
import torch.nn as nn
import torch.nn.functional as F

class ModelTanh(nn.Module):
    def __init__(self):
        super(ModelTanh, self).__init__()
        self.Conv2d_1a_3x3 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.Conv2d_2a_3x3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
    def forward(self, x):
        tanh = torch.tanh(self.Conv2d_1a_3x3(x))
        t1 = torch.tanh(self.Conv2d_2a_3x3(tanh))
        return t1
# Inputs to the model
x = torch.randn(1, 3, 32, 32)
