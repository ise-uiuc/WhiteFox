
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = torch.randn(3, 3)
    def forward(self, x1, x2, v1, v2):
        v3 = torch.mm(x1, x2) + self.inp
        v4 = v2 + v3
        return v4
# Inputs to the model
x1 = torch.randn(3, 3, requires_grad=True)
x2 = torch.randn(3, 3)
v1 = torch.randn(3, 3)
v2 = torch.randn(3, 3)
