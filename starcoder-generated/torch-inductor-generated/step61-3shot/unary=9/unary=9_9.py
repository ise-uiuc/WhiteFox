
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, (1, 6), stride=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1.add(3.)
        v3 = v2.clamp(min=0., max=6.)
        v4 = v3 / 6.
        return v4
# Inputs to the model
x1 = torch.randn(5, 3, 64, 64)
