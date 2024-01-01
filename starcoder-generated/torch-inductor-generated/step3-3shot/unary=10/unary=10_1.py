
import torch.nn.functional as F

class Model(torch.nn.Module):
    def forward(self, x1):
        v1 = F.linear(x1, __constant__, __constant__)
        v2 = v1 + 3
        v3 = F.relu6(v2)
        v4 = v3 / 6
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1, 224, 224)
