
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = torch.randn(3, 3)
    def forward(self, x1, x2, v0):
        v2 = torch.mm(x1 * x1, torch.mm(x2, x2))
        v1 = torch.mm(x1, x2) + self.inp
        return v2
# Inputs to the model
x1 = torch.randn(3, 3, requires_grad=True)
x2 = torch.randn(3, 3, requires_grad=True)
v0 = torch.randn(3, 3, requires_grad=True)
