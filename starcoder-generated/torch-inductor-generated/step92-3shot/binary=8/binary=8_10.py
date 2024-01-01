
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, padding=1, stride=1)
        self.conv2 = torch.nn.Conv2d(3, 16, 3, padding=1, stride=1)
        self.conv3 = torch.nn.Conv2d(3, 16, 3, padding=1, stride=2, dilation=1)
        self.conv4 = torch.nn.Conv2d(3, 16, 3, padding=3, stride=1, dilation=2)
        self.bn1 = torch.nn.BatchNorm2d(16)
    def forward(self, x1, x2, x3, x4):
        v1 = self.conv1(x1)
        v2 = self.conv2(x2)
        v3 = v1 + v2
        v5 = self.bn1(v3)
        v6 = v5.detach()
        v4 = v6 + v6
        return v4
# Inputs to the model
x1 = torch.randn(4, 3, 32, 32)
x2 = torch.randn(4, 3, 32, 32)
x3 = torch.randn(4, 3, 16, 16)
x4 = torch.randn(4, 3, 16, 16)
