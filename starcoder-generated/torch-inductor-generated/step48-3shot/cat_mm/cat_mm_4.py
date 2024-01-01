
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        x1 = x1.detach()
        x2 = x2.detach()
        b = []
        for loopVar1 in range(130):
            for loopVar2 in range(130):
                v1 = torch.mm(x1, x2)
                b.append(v1)
        return torch.stack(b, 1)
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3)
