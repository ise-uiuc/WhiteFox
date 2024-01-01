
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(int(input_size), int(hidden_size), bias=False)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - x1
        return v2

# Initializing the model
import numpy as np
m = Model()

# Inputs to the model
input_size = 37
hidden_size = 500000000
x1 = torch.randn(1, int(input_size))
