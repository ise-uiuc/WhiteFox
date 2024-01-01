
import torch.nn as nn
 
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 5)
 
    def forward(self, x):
        y_linear = self.linear(x)
        z = y_linear + y_linear
        return z

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 2)
