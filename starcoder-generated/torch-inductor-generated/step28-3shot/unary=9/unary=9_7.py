
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1, x2):
        v1 = self.conv(x1)
        v2 = 3 + v1
        v3 = v2.clamp(min=0, max=6)
        v4 = v3 / 6
        v5 = self.conv(x2)
        v6 = 4 + v5
        v7 = v6.clamp(min=-4, max=3)
        v8 = v7 / 3
        return v8
# Inputs to the model
x1 = torch.randn(5, 3, 28, 28)
x2 = torch.randn(1, 3, 256, 256)
