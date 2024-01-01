
import torch
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.split = torch.split(3, [1, 1])

    def forward(self, x1):
        v1, v3, v4 = self.split(x1)
        v2 = torch.cat([v3, v4])
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.tensor([1, 2, 3]).view(2, 1)
