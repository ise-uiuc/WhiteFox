
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = torch.mm(x1, x2)
        v2 = torch.mm(v1, v1)
        v3 = torch.mm(v2, v2)
        return torch.cat([v3, v3, v3], 1)
# Inputs to the model
x1 = torch.randn(64, 3)
x2 = torch.randn(64, 3)
