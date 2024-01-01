
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = []
        for loopVar1 in range(3):
            vi = torch.mm(x1, x2)
            v1.append(vi)
            v1.append(vi)
            v1.append(vi)
            v1.append(vi)
            v1.append(vi)
            v1.append(vi)
        v2 = []
        for loopVar2 in range(3):
            vi = torch.mm(x2, x1)
            v2.append(vi)
            v2.append(vi)
            v2.append(vi)
            v2.append(vi)
            v2.append(vi)
            v2.append(vi)
        v = torch.cat([v1, v2], 1)
        return v
# Inputs to the model
x1 = torch.randn(5, 2)
x2 = torch.randn(2, 5)
