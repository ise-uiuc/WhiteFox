
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = torch.mm(x1, x2)
        for loopVar1 in range(285):
            v1 = torch.mm(x1, x2)
        v2 = torch.mm(x1, x2)
        for loopVar1 in range(69):
            v2 = torch.mm(x1, x2)
            v2 = torch.mm(x1, x2)
        v3 = torch.mm(x1, x2)
        for loopVar1 in range(2):
            v3 = torch.mm(x1, x2)
            v3 = torch.mm(x1, x2)
            v3 = torch.mm(x1, x2)
        return torch.cat([v1, v2, v3], 1)
# Inputs to the model
x1 = torch.randn(7, 69)
x2 = torch.randn(69, 47)
