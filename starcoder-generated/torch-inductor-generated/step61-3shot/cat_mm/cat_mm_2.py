
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        v1 = torch.mm(x, y)
        v2 = torch.mm(x, y)
        v3 = torch.mm(x, y)
        return torch.cat([1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 7, 8], 0)
# Inputs to the model
x = torch.randn(5, 5)
y = torch.randn(5, 5)
