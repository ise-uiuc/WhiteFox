
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        for i in range(10):
            x1 += 1
        x2 = torch.randn(1, 2, 3)
        x3 = torch.rand_like(x1)
        x4 = x3 * torch.rand(1)
        x5 = x2*x4
        return x5

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        for i in range(10): x1 += 1
        x2 = torch.randn(1, 2, 3)
        x3 = torch.rand_like(x1)
        x4 = x3 * torch.rand(1)
        x5 = x2*x4
        for j in range(10): x1 += 1
        return x5
# Inputs to the model
x1 = torch.randn(1, 2, 2)
