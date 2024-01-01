
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v = torch.mm(x1, x2)
        for loopVar1 in range(5):
            v1 = torch.mm(x1, x2)
        for loopVar1 in range(50):
            v1 = torch.mm(x1, x2)
            v2 = torch.mm(x1, x2)
        for loopVar1 in range(10):
            v2 = torch.mm(x1, x2)
        return torch.cat([v, v1, v2], 1)
# Inputs to the model
x1 = torch.randn(4, 4)
x2 = torch.randn(4, 4)
