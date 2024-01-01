
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = torch.mm(x1, x2)
        t7 = torch.cat([v1, v1], 1)
        t9 = torch.cat([t7, v1], 1)
        t2 = torch.cat([t9, v1, v1, v1, v1, t9, v1, v1, t9], 1)
        t4 = torch.cat([t2, v1, v1, t2], 1)
        return torch.cat([t4, v1, v1, t4], 1)
# Inputs to the model
x1 = torch.randn(4, 2)
x2 = torch.randn(2, 4)
