
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = torch.mm(x1, x2)
        x3 = torch.randn(2, 6)
        v2 = torch.mm(x3, x2)
        v3 = torch.mm(x3, x2)
        return torch.cat([v1, v2, v3], 1)
# Inputs to the model
x1 = torch.randn(1, 3)
x2 = torch.randn(1, 2)
