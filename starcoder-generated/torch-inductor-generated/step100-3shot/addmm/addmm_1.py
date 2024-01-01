
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, y):
        v1 = torch.mm(x1, x2)
        v2 = v1 + y
        v3 = torch.mm(x1, x2)
        v4 = v2 + v3
        return v4
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3)
y = torch.randn(3, 3, requires_grad=True)
