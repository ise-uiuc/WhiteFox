
import torch.nn as nn
 
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 8)
 
    def forward(self, x):
        v1 = self.fc1(x)
        v2 = torch.clamp(min=0, max=6, v1 + 3)
        v3 = v2 / 6
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3)

output = m(x)

