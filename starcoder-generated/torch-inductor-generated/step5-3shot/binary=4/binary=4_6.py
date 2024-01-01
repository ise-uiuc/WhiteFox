
import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(20, 30)
 
    def forward(self, x):
        v1 = self.linear(x)
        y = v1 + y
        return y

# Initializing the model
m = Model()

# Inputs to the model
__input__ = torch.randn(1, 20)
