
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = torch.mm(x1, x2)
        for loopVar1 in range(457):
            v1 = torch.mm(x1, x2)
            v1 = torch.mm(x1, x2)
            v1 = torch.mm(x1, x2)
            v1 = torch.mm(x1, x2)
            v1 = torch.mm(x1, x2)
        return torch.cat([v1, v1, v1, v1, v1], 1)
# Inputs to the model
x1 = torch.randn(7, 7)
x2 = torch.randn(7, 5)
