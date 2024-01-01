
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=2, padding=1, dilation=1, groups=1)
        self.conv3x3 = torch.nn.Conv2d(3, 3, 3, stride=1, padding=1, dilation=1, groups=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv3x3(x1)
        return v1 * v2
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
