
import torch.nn.functional as F
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 64)

    def forward(self, x1):
        l1 = self.linear(x1)
        l2 = l1 + 3
        l3 = F.relu6(l2)
        l4 = l3 / 6
        return l4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(10, 64)
