
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = torch.rand_like(x1)
        x3 = torch.rand_like(x1)
        x5 = x2 - x3
        x6 = self.f(x1)
        x5 = x6 + x5
        return x5
# Inputs to the model
x1 = torch.randn(1, 2, 2)
