
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, x3, x4, x5, x6):
        v1 = x1 + x2
        v2 = x3 + x4
        v3 = x5 + x6
        return v1 + v2 + v3 + x5
# Inputs to the model
x1 = torch.randn(3, 3, requires_grad=True)
x2 = torch.randn(3, 3)
x3 = torch.randn(3, 3, requires_grad=True)
x4 = torch.randn(3, 3)
x5 = torch.randn(3, 3, requires_grad=True)
x6 = torch.randn(3, 3)
