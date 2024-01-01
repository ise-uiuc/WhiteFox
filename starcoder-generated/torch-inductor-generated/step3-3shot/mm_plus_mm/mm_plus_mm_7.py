
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, x3, x4, x5):
        v1 = torch.mm(x1, x2)
        v2 = torch.mm(x3, x4)
        v3 = v1 + v2
        v4 = torch.mm(torch.mm(x1, x5), x4) + torch.mm(x2, x5)
        v5 = v3 + v4
        return v5
# Inputs to the model
x1 = torch.randn(1, 3)
x2 = torch.randn(3, 3)
x3 = torch.randn(1, 3)
x4 = torch.randn(3, 4)
x5 = torch.randn(3, 4)
