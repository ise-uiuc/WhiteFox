
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = torch.mm(x1, x2)
        v2 = torch.mm(x1, v1)
        v3 = torch.mm(x1, v2)
        t1 = torch.cat([v1, v1, v1, v1, v1, v1, v1, v1], 1)
        t2 = torch.cat([v2, v2, v2, v2, v2, v2, v2, v2], 1)
        t3 = torch.cat([v3, v3, v3, v3, v3, v3, v3, v3], 1)
        return t1
# Inputs to the model
x1 = torch.randn(2, 3)
x2 = torch.randn(3, 2)
