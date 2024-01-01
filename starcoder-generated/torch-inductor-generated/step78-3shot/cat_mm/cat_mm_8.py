
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = torch.mm(x1, x2)
        t1 = torch.cat([v1, v1], 2)
        t2 = torch.cat([t1, t1, t1, t1, t1, t1, t1, t1, t1, t1, t1, t1, t1], 2)
        return torch.cat([t2, t2, t2, t2], 2)
# Inputs to the model
x1 = torch.randn(1, 64, 5, 5)
x2 = torch.randn(64, 1, 5, 5)
