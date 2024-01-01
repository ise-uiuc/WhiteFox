
import torch.nn as nn
 
class Model(nn.Module):
    def __init__(self):
      super().__init__()
      self.linear = nn.Linear(10, 10)
 
    def forward(self, x0):
      v1 = self.linear(x0)
      v2 = torch.sigmoid(v1)
      return v2

# Initializing the model
m = Model()

# Input to the model
x0 = torch.zeros(1, 10, 10)
