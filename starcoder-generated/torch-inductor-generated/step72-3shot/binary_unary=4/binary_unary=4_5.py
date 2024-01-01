
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(256, 256)
        self.other = other
 
    def forward(self, x1):
        v2 = self.other
        v1 = self.linear(x1)
        v3 = v1 + v2
        v4 = torch.nn.functional.relu(v3)
        return v4

# Initializing the model
import torch
m = Model(torch.randn(1, 256))

# Inputs to the model
x1 = torch.randn(1, 256)
