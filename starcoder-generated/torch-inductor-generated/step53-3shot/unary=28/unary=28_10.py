
class Model(torch.nn.Module):
    def __init__(self, min_value: float, max_value: float):
        super().__init__()
        self.linear = torch.nn.Linear(64, 1)
        self.min_value = min_value
        self.max_value = max_value
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        return v3

# Initializing the model
m = Model(6, 16)

import torch
m = torch.nn.Linear(2, 7)
input0 = torch.randn(1, 2)
v = m(input0)

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
