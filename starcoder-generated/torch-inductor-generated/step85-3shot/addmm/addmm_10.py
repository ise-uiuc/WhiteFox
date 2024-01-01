
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, a, b, c):
        v1 = torch.mm(a, x1)
        return v1 + c
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3)
a = torch.randn(3, 3, requires_grad=True)
b = torch.randn(3, 3, requires_grad=True)
c = torch.randn(3, 3, requires_grad=True)
