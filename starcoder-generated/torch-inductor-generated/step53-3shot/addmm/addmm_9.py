
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, v0, v3):
        v1 = torch.mm(x1, x2)
        v4 = v1 + v3
        return v0 + v1 + v4
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3)
v0 = torch.randn(3, 3, requires_grad=True)
v3 = torch.randn(3, 3, requires_grad=True)
