
import torch.nn as nn
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(100, 100)
        self.linearlayers = nn.Linear(100, 64)
    def forward(self, x):
        x = self.linearlayers(self.layers(x))
        return x
# Inputs to the model
x = torch.randn(2, 100)
