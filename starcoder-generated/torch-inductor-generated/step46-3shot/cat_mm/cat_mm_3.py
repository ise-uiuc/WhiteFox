
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = torch.mm(x1, x2)
        v2 = torch.mm(x1, x2)
        v = v1
        if int(torch.sum(v2)):
            for loopVar1 in range(4):
                v = torch.mm(x1, x2)
        return torch.cat([v] * 5, 1)
# Inputs to the model
x1 = torch.randn(2, 4)
x2 = torch.randn(4, 4)
