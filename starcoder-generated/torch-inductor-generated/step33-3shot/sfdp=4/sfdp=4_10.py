
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.wq = torch.nn.parameter.Parameter(torch.randn(1, 64, 56, 56, 8, 8))
    def forward_(self):
        x = self.wq + self.wq
        return x
x = Model()
x.eval()
print(x())
