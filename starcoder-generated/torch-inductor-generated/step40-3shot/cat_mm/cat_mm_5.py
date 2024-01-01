
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v = []
        for loopVar1 in range(6):
            v.append(torch.mm(x1, x2))
            v.append(torch.mm(x1, x2))
            v.append(torch.mm(x1, x2))
            v.append(torch.mm(x1, x2))
        return torch.cat(v, 1)
# Inputs to the model
x1 = torch.randn(3, 2)
x2 = torch.randn(2, 3)
