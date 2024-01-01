
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3, False)
        self.bn = torch.nn.BatchNorm1d(3, affine=False)
    def forward(self, x, inp):
        v1 = torch.mm(x, x)
        p = self.bn(None)
        self.bn.eval()
        p = self.bn(None)
        p = self.bn(None)
        p = self.bn(None)
        p = p + inp
        p = self.bn(None)
        p = self.bn(None)
        p = self.bn(None)
        v2 = torch.mm(v1, v1)
        self.bn.eval()
        v3 = self.bn(None)
        v3 = self.linear(inp)
        v3 = self.bn(v3)
        self.bn.eval()
        p = self.linear(p)
        p = self.bn(p)
        p = self.bn(p)
        p = self.bn(p)
        return (v1, v2, v3)
# Inputs to the model
x = torch.randn(3, 3)
inp = torch.randn(3, 3)
