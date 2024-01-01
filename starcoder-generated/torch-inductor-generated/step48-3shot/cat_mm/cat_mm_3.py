
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        x1 = x1.detach()
        x2 = x2.detach()
        v1 = torch.mm(x1, x2)
        v2 = torch.mm(x1, x2)
        v3 = torch.mm(x1, x2)
        t1 = torch.cat([v1, v2], 1)
        return torch.cat([t1, t1, t1], 1)
# Inputs to the model
x1 = torch.randn(1, 2)
x2 = torch.randn(2, 3)
