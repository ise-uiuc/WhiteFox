
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        t1 = torch.mm(x1, x2)
        return torch.cat([t1, t1], 1)

class Model2(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        t = torch.mm(x1, x2)
        if (t.numel() < 3):
            return torch.cat([t, t], 1)
        else:
            return torch.cat([t, t, t], 1)
# Inputs to the model
x1 = torch.randn(4, 4)
x2 = torch.randn(4, 4)
