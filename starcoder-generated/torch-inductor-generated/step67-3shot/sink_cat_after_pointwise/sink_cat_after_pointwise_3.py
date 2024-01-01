
from torch.nn import init
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = x + torch.ones_like(x)
        z = torch.cat([x, x]).requires_grad_()
        return z
# Inputs to the model
x = torch.randn(2, requires_grad=True)
