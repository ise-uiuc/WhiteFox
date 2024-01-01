
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = F.dropout2d(x1, p=0.5)
        x3 = torch.rand_like(x2)
        return x2
# Inputs to the model
x1 = torch.randn(1, 1, 2, 2)

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        x3 = F.dropout2d(x1, p=0.5)
        x4 = F.dropout2d(x2, p=0.5)
        x5 = torch.rand_like(x3)
        x6 = torch.rand_like(x4)
        return x5 + x6
# Inputs to the model
x1 = torch.randn(1, 1, 2, 2)
x2 = torch.randn(1, 1, 2, 2)
