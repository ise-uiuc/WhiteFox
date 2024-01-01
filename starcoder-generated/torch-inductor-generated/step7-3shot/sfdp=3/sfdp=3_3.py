
import torch.nn as nn
 
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(768, 768)
 
    def forward(self, x1, x2):
        q = self.proj(x1)
        k = self.proj(x2)
        v = self.proj(x3)
        o= torch.matmul(q, k.transpose(-2, -1))
        return o

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 768, 12, 13)
x2 = torch.randn(1, 768, 15, 16)
x3 = torch.randn(1, 768, 15, 16)
