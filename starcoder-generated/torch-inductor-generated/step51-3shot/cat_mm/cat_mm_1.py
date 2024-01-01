
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        v1 = torch.mm(x, y)
        return torch.cat([v1, v1, v1, v1], 1)
# Inputs to the model
x1 = torch.randn(1, 3)
x2 = torch.randn(3, 1)
