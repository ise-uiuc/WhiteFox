
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False)
    
    def forward(self, input_x, other=0):
        v1 = self.conv(input_x)
        v2 = v1 + other
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 32, 64, 64)
