
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = torch.randn(3, 3, requires_grad=True)
    def forward(self, x1, x2, b1):
        v1 = torch.mm(x1, b1)
        v2 = v1 + self.inp
        return v2
# Inputs to the model
x1 = torch.randn(3, 3, requires_grad=True)
x2 = torch.randn(3, 3, requires_grad=True)
b1 = torch.randn(3, 3, requires_grad=True)
