
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.gpool1 = torch.nn.AdaptiveAvgPool2d(1)
        self.conv2 = torch.nn.Conv2d(64, 128, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.bn1(v1)
        v3 = self.gpool1(v2)
        v4 = self.conv2(v3)
        v5 = v4 * 3
        v6 = v5 - 6
        v7 = v6 / -6
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
