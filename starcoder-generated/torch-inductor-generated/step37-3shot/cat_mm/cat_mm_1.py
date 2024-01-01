
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        for loopVar1 in range(10):
            v = torch.mm(x1, x2)
        for loopVar2 in range(10):
            v = torch.mm(x1, x2)
        for loopVar3 in range(10):
            v = torch.mm(x1, x2)
        return v
# Inputs to the model
x1 = torch.randn(1, 20)
x2 = torch.randn(20, 1)
