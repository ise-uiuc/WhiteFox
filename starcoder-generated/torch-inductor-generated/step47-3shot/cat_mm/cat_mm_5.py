
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        r1 = torch.roll(x1, [0, 1], 1)
        r2 = torch.roll(x2, [2, 1], 1)
        return torch.cat([r1, r2, r2, r1], 1)
# Inputs to the model
x1 = torch.randn(2, 2)
x2 = torch.randn(2, 2)
