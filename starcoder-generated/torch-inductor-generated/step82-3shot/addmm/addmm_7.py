
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, y):
        v1 = torch.mm(x1, y) + x2
        return torch.mm(x2, v1)
# Inputs to the model
x1 = torch.randn(3, 3, requires_grad=True)
x2 = torch.randn(3, 3, requires_grad=True)
v2 = torch.randn(3, 3, requires_grad=True)
y = torch.randn(3, 3)
