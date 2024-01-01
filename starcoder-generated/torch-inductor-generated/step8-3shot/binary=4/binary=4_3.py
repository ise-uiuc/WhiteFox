
import torch.nn as nn
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = nn.Linear(1, 1)
        self.linear_2 = nn.Linear(1, 1)
 
    def forward(self, x, other):
        v1 = self.linear_1(x)
        v2 = v1 + other
        return self.linear_2(v2)

# Initializing the model
m = Model()

# Inputs to the model
x = torch.tensor([[[-10.0]]])
other = torch.tensor([[[-8.0]]])
