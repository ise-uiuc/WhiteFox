
class Permuted(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x.permute(1, 0, 2, 3)
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.m1 = Permuted()
        self.m2 = Permuted()
    def forward(self, x1, x2):
        v1 = self.m1(x1)
        v2 = self.m2(x2)
        v3 = torch.bmm(v1, v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 2, 2, 2)
x2 = torch.randn(1, 2, 2, 2)
