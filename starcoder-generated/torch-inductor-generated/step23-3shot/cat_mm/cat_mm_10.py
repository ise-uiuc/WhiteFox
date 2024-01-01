
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        x2 = torch.cat([x2, x2, x2], 1)
        v1 = torch.mm(x1, x2)
        v2 = torch.mm(x1, x2)
        v3 = torch.mm(x1, x2)
        return torch.cat([v2, v3, v2, v1], 1)
# Inputs to the model
x1 = torch.randn(3, 5)
x2 = torch.randn(4, 5)
