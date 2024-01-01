
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v = torch.mm(x1, x2)
        for loopVar1 in range(int(4)):
            v = torch.mm(x1, x2)
        return torch.mm(v, v)
# Inputs to the model
x1 = torch.randn(2, 1)
x2 = torch.randn(1, 4)
