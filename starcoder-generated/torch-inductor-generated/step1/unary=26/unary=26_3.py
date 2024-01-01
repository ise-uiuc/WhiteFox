
import torch
class LeakyReLU(torch.nn.Module):
    def __init__(self, negative_slope: float = 0.2, inplace: bool = False):
        super().__init__()
        self.negative_slope = negative_slope
        self.inplace = inplace
 
    def forward(self, x):
        v1 = x > 0
        v2 = torch.where(v1, x, x * self.negative_slope)
        return v2

# Initializing the model
m = LeakyReLU(negative_slope=0.5)

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
