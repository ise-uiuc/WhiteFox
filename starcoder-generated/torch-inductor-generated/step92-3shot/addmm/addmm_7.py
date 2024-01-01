
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = torch.randn(3, 3)
    def forward(self, param, x1, v0):
        v1 = torch.mm(x1, param)
        v2 = v1 + self.inp
        return v2
# Inputs to the model
param = torch.randn(3, 3)
x1 = torch.randn(3, 3)
v0 = torch.randn(3, 3)
