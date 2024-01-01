
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x3):
        v1 = torch.mm(x1, x2)
        t1 = torch.cat([v1, v1], 1)
        return torch.cat([t1, x3], 1)
# Inputts to the model
x1 = torch.randn(4, 2)
x2 = torch.randn(2, 4)
x3 = torch.randn(4, 4)
