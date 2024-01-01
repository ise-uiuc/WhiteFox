
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = torch.cat([x1] * 1, 1)
        v2 = torch.cat([x1] * 2, 1)
        v3 = torch.cat([x1] * 3, 1)
        v4 = torch.cat([x1] * 4, 1)
        v5 = torch.cat([v1] * 2, 1)
        v6 = torch.cat([v1, v2, v3], 1)
        v7 = torch.cat([v5, v2, x1, v5], 1)
        v8 = torch.cat([v7] * 5, 1)
        return torch.cat([v6, v1, v2, v3, v4, v5, v6, v7, v8], 1)
# Inputs to the model
x1 = torch.randn(10, 10, 1)
