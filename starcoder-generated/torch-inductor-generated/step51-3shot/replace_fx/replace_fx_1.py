
class Model(torch.nn.Module):
    def __init__(self, p2):
        super().__init__()
        self.register_buffer('self.p2', torch.tensor(p2))
        self.bn = torch.nn.BatchNorm1d(2)
    def forward(self, x1):
        x2 = self.bn(x1)
        x3 = torch.rand_like(x2)
        return x3
p2 = 0.7
# Inputs to the model
x1 = torch.randn(3, 3)
