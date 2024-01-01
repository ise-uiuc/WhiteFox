
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(30, 32, 3, stride=2, padding=1, groups=2)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.flip(v1, [2])
        return v2
# Inputs to the model
x1 = torch.randn(6, 30, 64, 64)
