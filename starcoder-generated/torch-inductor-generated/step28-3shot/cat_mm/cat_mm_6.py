
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        v1 = torch.mm(x, y)
        v2 = torch.mm(x, y)
        return torch.cat([v1] * 3, 1), torch.cat([v2] * 3, 1), torch.cat([v1] * 2, 1), torch.cat([v2] * 2, 1)
# Inputs to the model
x = torch.randn(2, 2)
y = torch.randn(2, 2)
