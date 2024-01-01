
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, x2, x3, x4, x5):
        v1 = torch.mm(x, x2)
        v2 = torch.mm(x3, x)
        v3 = torch.mm(x4, x5)
        v4 = v1 - v2 + v3
        v5 = torch.mm(torch.mm(x2, x2), v4)
        return torch.mm(v5, x3)
# Inputs to the model
x = torch.randn(3, 3)
x2 = torch.randn(3, 3)
x3 = torch.randn(3, 3)
x4 = torch.randn(3, 3)
x5 = torch.randn(3, 3)
# model ends