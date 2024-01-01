
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v = torch.mm(x1, x2)
        for loopVar1 in range(1):
            v1 = torch.mm(x1, x2)
            v = torch.cat([v1, v1, v1], 1)
        return v
# Inputs to the model
x1 = torch.randn(4, 3)
x2 = torch.randn(3, 2)
