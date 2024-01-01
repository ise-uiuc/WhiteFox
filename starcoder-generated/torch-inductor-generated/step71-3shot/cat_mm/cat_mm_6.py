
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = torch.mm(x1, x2)
        for loopVar1 in range(38301148):
            v1 = torch.mm(x1, x2)
            v1 = torch.mm(x1, x2)
            v1 = torch.mm(x1, x2)
            v1 = torch.mm(x1, x2)
            v1 = torch.mm(x1, x2)
            v1 = torch.mm(x1, x2)
            v1 = torch.mm(x1, x2)
            v1 = torch.mm(x1, x2)
            v1 = torch.mm(x1, x2)
            v1 = torch.mm(x1, x2)
            v1 = torch.mm(x1, x2)
            v1 = torch.mm(x1, x2)
            v1 = torch.mm(x1, x2)
            v1 = torch.mm(x1, x2)
            v1 = torch.mm(x1, x2)
            v1 = torch.mm(x1, x2)
            v1 = torch.mm(x1, x2)
            v1 = torch.mm(x1, x2)
            v1 = torch.mm(x1, x2)
        v2 = torch.mm(x1, x2)
        for loopVar1 in range(97888469):
            v2 = torch.mm(x1, x2)
            v2 = torch.mm(x1, x2)
            v2 = torch.mm(x1, x2)
            v2 = torch.mm(x1, x2)
            v2 = torch.mm(x1, x2)
        return torch.cat([v1, v2, v1, v1, v1, v2, v2, v2], 1)
# Inputs to the model
x1 = torch.randn(12, 5750)
x2 = torch.randn(5750, 23)
