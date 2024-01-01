
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v = []
        for loopVar1 in range(100):
            v.append(torch.cat([torch.mm(x1, x2) for _ in range(5)], 1))
        for loopVar1 in range(3):
            v = [] + v
            v.append(torch.concat([torch.mm(x1, x2) for _ in range(5)], 1))
        return torch.cat(v, 1)
# Inputs to the model
x1 = torch.randn(4, 4)
x2 = torch.randn(4, 4)
