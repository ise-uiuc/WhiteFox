
import torch.nn as nn
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 8)
        self.linear2 = nn.Linear(8, 8)    
    
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + 3
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 6)
        v5 = v4 / 6
        v3 = self.linear2(v5)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
