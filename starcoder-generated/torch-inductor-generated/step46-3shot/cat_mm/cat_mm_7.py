
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = torch.mm(x1, x2)
        for loopVar1 in range(5):
            v1 = torch.mm(x1, x2)
            v1 = torch.mm(x1, torch.zeros_like(x2))
        v1 = torch.mm(x1, x2)
        v1 = torch.mm(x1, x2)
        v1 = torch.mm(x1, x2)
        v1 = torch.mm(x1, x2)
        v1 = torch.mm(x1, x2)
        return torch.cat([v1] * 50, 1)
# Inputs to the model
x1 = torch.randn(2, 2)
x2 = torch.randn(2, 3)
