
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, x3, x4, x5, other=None):
        v1 = torch.cat((x1, x2), 1)
        v2 = torch.cat((x3, x3), 1)
        v3 = torch.cat((v1, v2), 1)
        if other == None:
            other = (v3 + v4) * x5
        v4 = v3 + other
        return v4
# Inputs to the model
x1 = torch.randn(2, 1, 64, 64)
x2 = torch.randn(2, 1, 64, 64)
x3 = torch.randn(2, 2, 64, 64)
x4 = torch.randn(2, 2, 64, 64)
x5 = torch.randn(768, 64, 64)
