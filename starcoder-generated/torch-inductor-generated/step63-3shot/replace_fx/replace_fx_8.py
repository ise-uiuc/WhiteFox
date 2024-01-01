
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        xo = F.dropout(x2)
        return xo
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        F.dropout(x2)
        return x2
# Inputs to the model
x1 = torch.randn(1, 2, 3)
x2 = torch.randn(1, 2, 3)

