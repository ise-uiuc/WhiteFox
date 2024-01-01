
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = torch.randn(3, 3, requires_grad=True)
    def forward(self, x1, x2, v0):
        v1 = torch.mm(x1, x2)
        v2 = v1 + self.inp
        v3 = torch.mm(x1, x2)
        v4 = v2 + v3
        return v4 + v0
# Inputs to the model
x1 = torch.randn(3, 3, requires_grad=True)
x2 = torch.randn(3, 3, requires_grad=True)
v0 = torch.randn(3, 3)
