
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, x3, x4, x5, x6):
        v1 = torch.mm(x3, x6)
        v2 = torch.mm(x3, x6)
        v3 = torch.mm(x3, x6)
        return torch.cat([v2, v3, v3, v3, v3, v2, v2, v1], 1)
# Inputs to the model
x1 = torch.randn(1, 4)
x2 = torch.randn(1, 4)
x3 = torch.randn(1, 4)
x4 = torch.randn(1, 4)
x5 = torch.randn(1, 4)
x6 = torch.randn(1, 4)
