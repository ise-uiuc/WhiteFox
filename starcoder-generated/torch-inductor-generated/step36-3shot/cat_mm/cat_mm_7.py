
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        if torch.cat([torch.mm(x1, x2)], 1) > 0:
            for loopVar2 in range(3):
                for loopVar1 in range(5):
                    v = torch.mm(x1, x2)
            return torch.cat(v, 1)
        else:
            v1 = torch.mm(x1, x2)
            return torch.cat([v1, v1], 1)
# Inputs to the model
x1 = torch.randn(5, 5)
x2 = torch.randn(5, 5)
