
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(100, 1)
 
    def forward(self, x1):
        v1 = self.fc(x1)
        v2 = torch.sigmoid(v1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 100)
