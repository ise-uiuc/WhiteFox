
import torch
from torch.nn.functional import clip

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x1):
        v1 = torch.squeeze(x1)
        v2 = torch.transpose(v1, 0, -1)
        v3 = torch.squeeze(x1)
        v4 = torch.transpose(v3, 0, -1)
        v5 = torch.transpose(v2, 1, 0)
        v6 = torch.transpose(v4, 1, 0)
        v7 = torch.matmul(v5, v6)
        return v7

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(32, 102, 307, 307)
