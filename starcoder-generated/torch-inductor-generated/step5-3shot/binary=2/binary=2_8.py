
# Import torch and numpy.
import torch
import numpy as np

# Create the model.
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x, y):
        v = self.conv(x)
        w = y[..., None, None]
        v2 = v - w
        return v2

# Inputs to the model.
x1 = torch.randn(1, 3, 64, 64)
x2 = np.array([-0.2, 0.6, -2.3])
# Model output.
z = Model()(x1, x2)
