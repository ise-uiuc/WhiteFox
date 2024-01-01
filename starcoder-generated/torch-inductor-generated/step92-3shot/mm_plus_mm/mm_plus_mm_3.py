
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, x3, x4, x5):
        t1 = torch.mm(x1, x2)
        t2 = torch.mm(x3, x4)
        t3 = torch.mm(x5, x1)
        return t1 + t2 + t3
# Inputs to the model
x1 = torch.randn(64, 64)
x2 = torch.randn(64, 64)
x3 = torch.randn(64, 64)
x4 = torch.randn(64, 64)
x5 = torch.randn(64, 64)
