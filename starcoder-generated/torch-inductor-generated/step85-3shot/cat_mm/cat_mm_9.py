
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        for loopVar9 in range(2):
            v2 = torch.cat([torch.mm(x1, x2)], 1)
        return v2
# Inputs to the model
x1 = torch.randn(2, 2)
x2 = torch.randn(2, 1)
