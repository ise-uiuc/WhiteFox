
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        l2 = torch.rand_like(x1, dtype=torch.float64)
        z1 = torch.sin(torch.rand_like(x1) - torch.randn()) + torch.rand_like(x1)
        return z1 + l2 - x1
# Inputs to the model
x2 = torch.randn(1)
