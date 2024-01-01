
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = torch.nn.Linear(3, 8)
 
    def forward(self, x1, other):
        v1 = self.linear_1(x1)
        v2 = v1 + other
        v3 = torch.relu(v2)
        return v3

# Initializing the model
from torch_utils import ModuleWithIntermediateFeatures
m = Model()
m = ModuleWithIntermediateFeatures(m)

# Inputs to the model
x1 = torch.randn(1, 3)
x2 = torch.randn(1, 3)
