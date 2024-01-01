
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v = []
        for loopVar1 in range(3):
            v.append(torch.mm(x1, x2))
        for loopVar2 in range(5):
            v.append(torch.mm(x1, x2))
        v.append(torch.mm(x1, x2))
        return torch.cat(v, 2)
# Inputs to the model
x1 = torch.randn(2, 5)
x2 = torch.randn(5, 2)
