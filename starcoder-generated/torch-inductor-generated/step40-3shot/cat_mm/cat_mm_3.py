
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v = []
        for loopVar1 in range(8):
            v.append(torch.mm(x1, x2))
            v.append(torch.mm(x1, x2))
            v.append(torch.mm(x1, x2))
            v.append(torch.mm(x1, x2))
        return torch.cat(v, 2)
# Inputs to the model
x1 = torch.randn(3, 4, 5)
x2 = torch.randn(5, 3)
