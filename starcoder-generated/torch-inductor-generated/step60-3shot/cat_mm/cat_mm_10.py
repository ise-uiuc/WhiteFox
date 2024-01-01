
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = torch.mm(x1, x2)
        v2 = torch.mm(x1, x2)
        v3 = torch.cat([x1, x1, x2, x2, x2], 0)
        v4 = torch.cat([v1, v2, v3], 1)
        return v4
# Inputs to the model
x1 = torch.randn(5, 5)
x2 = torch.randn(5, 5)
