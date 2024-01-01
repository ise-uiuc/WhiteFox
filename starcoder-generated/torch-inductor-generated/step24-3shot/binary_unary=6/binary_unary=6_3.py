
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 84)
 
    def forward(self, x):
        v1 = self.fc1(x)
        v2 = v1 - 0
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = Model(100)

# Inputs to the model
x = torch.randn(10, 100)
