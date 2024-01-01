
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, x3, x4, x5):
        v1 = torch.mm(x1, x2)
        v2 = torch.mm(x3, x4)
        v3 = torch.mm(v1, v2)
        v4 = v3 + x5
        return v4
# Inputs to the model
x1 = torch.randn(5, 35)
x2 = torch.randn(35, 30)
x3 = torch.randn(5, 4)
x4 = torch.randn(4, 35)
x5 = torch.randn(5)
