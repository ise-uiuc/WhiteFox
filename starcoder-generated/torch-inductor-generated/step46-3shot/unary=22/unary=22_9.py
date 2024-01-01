

import torch
import torch.nn as nn

class Model(nn.Module): 
    def __init__(self, in_features = 2048, out_features = 1000):
        super().__init__()
     
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.tanh(v1)
        return v2

# Initializing the model
m = Model()

# Input to the model
x1 = torch.randn(32, 2048)
