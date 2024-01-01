
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1682, 7, 3, stride=2, padding=0, dilation=1, groups=1, bias=True)
        self.conv2 = torch.nn.Conv2d(7, 99, 7, stride=3, padding=0, dilation=1, groups=1)
        self.conv3 = torch.nn.Conv2d(99, 91, 1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv4 = torch.nn.Conv2d(91, 33, 5, stride=2, padding=0, dilation=1, groups=1)
    def forward(self, x):
        negative_slope = -0.4993094
        v1 = self.conv1(x)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = self.conv4(v3)
        v5 = v4 > 0
        v6 = v4 * negative_slope
        v7 = torch.where(v5, v4, v6)
        return v7
# Inputs to the model
x1 = torch.randn(6, 1682, 476, 701)
