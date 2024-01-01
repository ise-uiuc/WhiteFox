
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        t1 = torch.mm(x2, x1)
        v2 = torch.mm(x1, x2)
        t2 = torch.cat([v2, v2], 0)
        return torch.cat([v2, v2, t2, t2, t2, t2], 1)
# Inputs to the model
x1 = torch.randn(8, 2)
x2 = torch.randn(2, 8)
