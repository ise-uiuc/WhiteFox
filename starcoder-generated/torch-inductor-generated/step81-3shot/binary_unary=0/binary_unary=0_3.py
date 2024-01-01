
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 7, stride=1, padding=3, groups=3)
    def forward(self, x1, x2):
        v1 = self.conv(x1)
        v2 = v1 + x1
        v3 = torch.sin(v2)
        v4 = self.conv(x2)
        v5 = v4 + x2
        v6 = torch.cos(v5)
        v7 = v3 + v6
        v8 = torch.sin(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
x2 = torch.randn(1, 1, 64, 64)
