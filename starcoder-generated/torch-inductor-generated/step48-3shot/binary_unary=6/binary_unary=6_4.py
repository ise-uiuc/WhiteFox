
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 64)
 
    def forward(self, x1, x2):
        v1 = self.linear(x2)
        v2 = v1 - x1
        v3 = torch.nn.functional.relu(v2)   # use torch.relu instead
        return v3, v2

# Initializing the model
m = Model()
import torch.nn.functional as F   # or import operator and use torch.relu(x2)
torch.manual_seed(1)

# Input to the model
x1 = torch.randn(1, 8)
x2 = torch.randn(1, 8)
__output_1, __output_2__ = m(x1, x2)

# Inputs to the model
torch.manual_seed(1)
x1 = torch.randn(1, 8)
__output_1, __output_2__ = m(x1, x2)

