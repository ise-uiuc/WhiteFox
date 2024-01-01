
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v = []
        for _ in range(10):
            v.append(torch.mm(x1, x2))
        tt = torch.cat(v, 1)
        return torch.cat([tt, tt], 0)
# Inputs to the model
x1 = torch.randn(10, 10)
x2 = torch.randn(10, 10)
