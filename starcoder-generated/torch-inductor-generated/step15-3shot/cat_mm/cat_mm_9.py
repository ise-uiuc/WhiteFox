
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = torch.cat([x1, x1, x1], 1)
        v2 = torch.cat([x2, x2, x2], 1)
        v3 = torch.cat([v1, v1, v1, v1, v1], 1)
        v4 = torch.cat([v1, v1, v1, v1, v1], 1)
        return torch.div(v3 - v4, v4)
# Inputs to the model
x1 = torch.randn(1, 3)
x2 = torch.randn(1, 3)
