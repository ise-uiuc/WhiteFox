
import math
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 16)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 * math.clamp(min=0, max=6.0, v1 + 3)
        v3 = v2 / 6.0
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 16)
