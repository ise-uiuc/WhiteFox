
class Model(torch.nn.Module):
    def __init__(self, n: int):
        super().__init__()
        self.in_features = n
    def forward(self, x):
        y = x.view(x.shape[0], -1)
        y = torch.tanh(y)
        z1 = torch.tanh(y)
        z2 = torch.tanh(z1)
        z3 = torch.tanh(z2)
        y = torch.cat([z1, z2, z3], dim=1) if self.in_features > 1 in (1, 10, 100, 1000) else y.view(x.shape[0], -1)
        return y
# Inputs to the model
n = 6
x = torch.randn(2, n, 4)
