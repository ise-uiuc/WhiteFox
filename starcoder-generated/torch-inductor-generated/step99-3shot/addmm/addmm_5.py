
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, x3, x4, x5, x6):
        v1 = torch.mm(x1, x6)
        v2 = torch.mm(x3, x4)
        v3 = torch.mm(x5, x2)
        v4 = v1+v2+v3
        return v4
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3)
x3 = torch.randn(3, 3)
x4 = torch.randn(3, 3)
x5 = torch.randn(3, 3)
x6 = torch.randn(3, 3)
