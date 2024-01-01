
import torch._ops
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 2)
        self.layers_2 = nn.Linear(2, 2)
        self.cat = lambda x, dim: torch._ops.aten.cat([x], di) # Concatenation
    def forward(self, x):
        x = self.layers(x)
        x = self.layers_2(x)
        x = self.cat([x, x], 1)
        x = torch.flatten(x, start_dim=1)
        return x
# Inputs to the model
x = torch.randn(2, 2)
