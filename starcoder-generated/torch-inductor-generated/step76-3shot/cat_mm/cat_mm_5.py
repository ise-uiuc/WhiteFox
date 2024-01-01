
class Model(torch.nn.Module):
    def __init__(self, b):
        super().__init__()
        self.b = b
    def forward(self, x1, x2):
        v = x1
        for _ in range(self.b):
            v = torch.mm(v, x2)
        return torch.cat([v, v, v, v], 1)
b = 1
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3)
