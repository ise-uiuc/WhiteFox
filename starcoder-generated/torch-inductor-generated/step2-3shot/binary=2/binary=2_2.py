
import torch
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
 
    def forward(self, x1, x2):
        v1 = self.conv(x1)
        v2 = v1 - x2
        return v2

# Initializing the model
# The size of the randomly generated 'other' tensor must match the size of the output of the convolution
other = torch.randn(1, 8, 64, 64)
m = Model()

# Inputs to the model (The size of the second input tensor must match the size of the output of the convolution)
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 8, 64, 64)
