
import torch.nn as nn
 
class Model(nn.Module):
    def __init__(self):
        super().__init__()
  # Linear transform: 160x16 -> 16x5
  # 160 = (16 + 12 - 8 + 0) * 5
  #     = 5 * 16 + 12 * 2 + 4
  #     = (16 * 5) + (12 * 4) + (4 * 2)
  #     = (16 * 5) + (48 * 100) + (4 * 2)
  #     = 80 + 48 + 16
        self.linear = torch.nn.Linear(16, 5, bias=False)
 
    def forward(self, x):
        v = self.linear(x)
        return v

# Initializing the model
m = Model()

# Inputs to the model
# The sizes of input tensors are not necessarily the sizes of output tensors
x = torch.randn(1, 16, 16, 16)
