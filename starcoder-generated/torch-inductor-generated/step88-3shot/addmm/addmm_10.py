
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = torch.randn(3, 3)
    def forward(self, x1, x2, x3):
        x1 = x1 + x2
        x2 = x1 + self.inp
        x3 = x3 + self.inp
        v1 = x1 + x2
        v2 = torch.mm(v1, x1)
        return x3 + v2
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3)
x3 = torch.randn(3, 3)
