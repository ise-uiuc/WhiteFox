
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x4, x2, v0):
        v1 = torch.mm(x1, x4)
        v3 = torch.mm(x2, x1)
        v2 = v1 + v3
        return v2 + v0
# Inputs to the model
x1 = torch.randn(3, 2)
x2 = torch.randn(3, 2, requires_grad=True)
x4 = torch.randn(3, 2, requires_grad=True)
v0 = torch.randn(3, 2)
