
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2, 2))
    def forward(self, x1):
        x2 = self.weight * x1
        x3 = torch.rand_like(x2)
        return x3
# Inputs to the model
x1 = torch.randn(2, 2, 2)
