
import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3, x4, x5):
        c1 = torch.cat([x1, x2, x3, x4, x5], dim=1)
        c2 = c1[:, c1.shape[1] : :]
        c3 = c2[:, 0:1024]
        out = torch.cat([c1, c3], dim=1)
        return out

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1024, 1024)
x2 = torch.randn(1, 1024, 1024)
x3 = torch.randn(1, 1024, 1024)
x4 = torch.randn(1, 1024, 1024)
x5 = torch.randn(1, 1024, 1024)
