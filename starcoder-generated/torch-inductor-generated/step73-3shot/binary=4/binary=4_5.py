
import torch.nn as nn
 
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(3, 6)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + x1
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 3)
other = torch.randn(1, 3)
