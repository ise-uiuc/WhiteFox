
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, x3):
        v = torch.mm(x1, x2)
        v = torch.mm(x1, x2)
        for _ in range(100):
            v = torch.mm(x1, x2)
            v = torch.mm(x1, x2)
        v = torch.mm(x1, x2)
        v = torch.mm(x1, x2)
        return torch.cat([v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v], 1)
# Inputs to the model
x1 = torch.randn(256, 256)
x2 = torch.randn(256, 256)
x3 = torch.randn(256, 256)
