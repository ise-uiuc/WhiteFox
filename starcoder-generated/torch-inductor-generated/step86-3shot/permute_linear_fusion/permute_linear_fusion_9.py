
import torch
class ModuleB(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 3)
    def forward(self, input):
        return self.linear(input)
import torch

class ModuleA(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mod_b = ModuleB()
    def forward(self, input):
        v1, v2 = torch.chunk(input, chunks= 2, dim = 1)
        return self.mod_b(torch.stack([v1, torch.empty_like(v2)], dim=1))

# Inputs to the model
x1 = torch.randn(1, 6, 2)
