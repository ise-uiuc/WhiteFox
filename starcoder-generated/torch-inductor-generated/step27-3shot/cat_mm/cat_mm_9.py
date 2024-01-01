
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = torch.mul(x1, x2)
        v2 = torch.cat(v1, 1)
        v3 = torch.cat([v2, v2, v2, v2], 1)
        return torch.mul(x1, v3)
# Inputs to the model
x1 = torch.randn(2, 1)
x2 = torch.randn(1, 4)
