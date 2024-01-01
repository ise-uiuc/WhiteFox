
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(24, 192, 1, padding=0, stride=1, dilation=1, groups=3, bias=True)
        self.conv2 = torch.nn.Conv2d(24, 192, 1, padding=0, stride=1, dilation=1, groups=3, bias=True)
        self.conv3 = torch.nn.Conv2d(24, 192, 1, padding=0, stride=1, dilation=1, groups=3, bias=True)
    def forward(self, x1, x2, x3):
        v1 = self.conv1(x1)
        v2 = self.conv2(x2)
        v3 = self.conv3(x3)
        v4 = v1 + v2 + v3
        v5 = torch.relu(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 24, 64, 64)
x2 = torch.randn(1, 24, 64, 64)
x3 = torch.randn(1, 24, 64, 64)
