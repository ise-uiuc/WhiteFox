
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v = [torch.mm(x1, x2)]
        v.append(torch.mm(x1, x2))
        v.append(torch.mm(x1, x2))
        for loopVar1 in range(100):
            v = torch.cat(v, 1)
            v = torch.split(v, 1, 1)
            v.append(torch.split(v, 1, 1)[-1])
        return v
# Inputs to the model
x1 = torch.randn(2, 3)
x2 = torch.randn(2, 3)
