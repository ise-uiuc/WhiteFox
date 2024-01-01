
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v = []
        x1 = torch.mm(x1, x2)
        v.append(x1)
        x1 = x1 + 1
        v.append(x1)
        for loopVar1 in range(6):
            v.append(x1)
            x1 = torch.mm(x1, x2)
            v.append(x1)
            x1 = x1 + 1
            v.append(x1)
        return torch.cat(v, 1)
# Inputs to the model
x1 = torch.randn(2, 2)
x2 = torch.randn(2, 2)
