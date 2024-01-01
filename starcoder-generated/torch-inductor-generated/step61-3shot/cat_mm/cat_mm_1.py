
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        v1 = torch.mm(x, y)
        v2 = torch.mm(x, y)
        v3 = torch.mm(x, y)
        return torch.cat([v1, v2, v3], 0)
# Inputs to the model
x = torch.randn(2, 2)
y = torch.randn(3, 2)
