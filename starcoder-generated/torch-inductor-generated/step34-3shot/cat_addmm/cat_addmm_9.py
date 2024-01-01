
import torch.nn.functional as F
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 4)
    def forward(self, x):
        x = F.linear(x, self.layers.weight, self.layers.bias)
        return x
# Inputs to the model
x = torch.randn(2, 2)
