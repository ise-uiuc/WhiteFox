
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 6, 1, stride=1, groups=2)
        self.conv2 = torch.nn.Conv2d(6, 6, 1, stride=1, groups=2)
        self.relu6 = torch.nn.ReLU6()
        self.conv3 = torch.nn.Conv2d(6, 6, 3, stride=1, padding=1, dilation=3)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv2(v1)
        v3 = self.relu6(v2)
        return self.conv3(v3)
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
