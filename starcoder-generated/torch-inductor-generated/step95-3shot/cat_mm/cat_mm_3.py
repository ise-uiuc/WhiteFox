
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, x3, x4):
        t2 = torch.mm(x2, x3)
        t1 = torch.mm(x1, x2)
        t = torch.cat([t1, t1, t1, t1, t1, t1, t1, t1, t1, t1, t1, t1], 1)
        return torch.cat([t, t, t, t, t, t, t, t, t, t, t, t, t, t, t, t, t, t, t1], 1)
# Inputs to the model
x1 = torch.randn(2, 3)
x2 = torch.randn(2, 3)
x3 = torch.randn(2, 3)
x4 = torch.randn(2, 3)
