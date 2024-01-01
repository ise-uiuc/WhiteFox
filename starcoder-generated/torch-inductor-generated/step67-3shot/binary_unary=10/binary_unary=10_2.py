
import torch.nn as nn
model = nn.Sequential(
    nn.Linear(10, 16, bias=True),
    nn.ReLU(),
    nn.Linear(16, 10, bias=True))

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 10)
