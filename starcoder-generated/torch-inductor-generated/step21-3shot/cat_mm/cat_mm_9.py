
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = torch.mm(x1, x2)
        v1 = torch.cat([v1, v1, v1, v1, v1], 0)
        v2 = torch.cat([v1, v1, v1, v1, v1], 1)
        v3 = torch.cat([v1, v1, v1, v1, v1], 2)
        v4 = torch.mm(v1, x2)
        v4 = torch.cat([v1, v1, v1, v1, v1], 0)
        return torch.cat([v4, v3, v2, v1, v4, v3, v2, v1], 0)
# Inputs to the model
x1 = torch.randn(1, 1)
x2 = torch.randn(1, 3)
