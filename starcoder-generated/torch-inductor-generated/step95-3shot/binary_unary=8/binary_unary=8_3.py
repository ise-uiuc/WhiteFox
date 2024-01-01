
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 192, 1, padding=0, stride=1, dilation=1, groups=1, bias=True)
    def forward(self, x1, x2, x3, x4, x5, x6):
        v1 = self.conv(x1)
        v2 = self.conv(x2)
        v3 = self.conv(x3)
        v4 = self.conv(x4)
        v5 = self.conv(x5)
        v6 = self.conv(x6)
        v7 = v1 + v2 + v3 + v4 + v5 + v6
        v8 = torch.relu(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 16, 16, 16)
x2 = torch.randn(1, 16, 16, 16)
x3 = torch.randn(1, 16, 16, 16)
x4 = torch.randn(1, 16, 16, 16)
x5 = torch.randn(1, 16, 16, 16)
x6 = torch.randn(1, 16, 16, 16)
