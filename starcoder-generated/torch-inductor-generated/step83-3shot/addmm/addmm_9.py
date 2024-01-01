
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, v0, v1):
        v2 = torch.mm(x1, x2)
        v3 = v2 + v1
        v4 = torch.mm(v0, v1)
        return v3 * v4
# Inputs to the model
x1 = torch.randn(3, 3, requires_grad=True)
x2 = torch.randn(3, 3, requires_grad=True)
v0 = torch.randn(3, 3)
v1 = torch.randn(3, 3)
