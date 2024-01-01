
from torch.nn import Linear
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = Linear(364, 392)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        return v2
# Inputs to the model
x1 = torch.randn(1, 364, 2)
