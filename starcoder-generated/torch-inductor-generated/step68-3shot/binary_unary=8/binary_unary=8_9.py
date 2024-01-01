
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 64, 3, stride=1, padding=1, groups=1, dilation=1)
        self.conv2 = torch.nn.Conv2d(64, 1, 3, stride=1, padding=1)
        self.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.avgpool(v1 + v2)
        v4 = torch.relu(v3)
        return v4
# Inputs to the model
x1 = torch.randn(257, 1, 64, 64)
