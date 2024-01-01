
import torch

class Model(torch.nn.Sequential):
    def __init__(self):
        super().__init__()
        self.add_module('linear', torch.nn.Linear(3, 8, bias=False))
        self.add_module('tanh', torch.nn.Tanh())

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
