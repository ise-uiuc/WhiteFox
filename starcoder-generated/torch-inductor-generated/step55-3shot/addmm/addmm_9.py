
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, x3, x4, x5, x6, x7, x8):
        v1 = torch.mm(torch.mm(x2, x3), x1)
        v2 = v1 + x4
        return v2
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3)
x3 = torch.randn(3, 3)
x4 = torch.randn(3, 3)
x5 = torch.randn(3, 3)
x6 = torch.randn(3, 3)
x7 = torch.randn(3, 3)
x8 = torch.randn(3, 3)
