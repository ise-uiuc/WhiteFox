
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x1 = x1 * x1
        x2 = torch.rand_like(x1)
        return x2
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        return F.dropout(x1, p=0.4)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
