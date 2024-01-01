
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 10, 1, padding=0, stride=1, dilation=1, groups=2,bias=True)
    def forward(self, x1, x2, x3):
        v1 = self.conv(x1)
        v2 = self.conv(x2)
        v3 = self.conv(x3)
        v4 = v1 + v2 + v3
        v5 = torch.relu(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 128, 128)
x2 = torch.randn(1, 3, 128, 128)
x3 = torch.randn(1, 3, 128, 128)
