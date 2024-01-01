
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 1, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.conv2 = torch.nn.Conv2d(64, 64, 1, stride=1, padding=0)
        self.avgpool1 = torch.nn.AvgPool2d(3, stride=2, padding=[2, 2], ceil_mode=True)
        self.conv3 = torch.nn.Conv2d(64, 64, 1, stride=1, padding=1)
        self.avgpool2 = torch.nn.AvgPool2d(3, stride=2, padding=[2, 2], ceil_mode=True)
        self.conv4 = torch.nn.Conv2d(64, 64, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.bn1(v1)
        v3 = self.conv2(v2)
        v4 = self.avgpool1(v3)
        v5 = self.conv3(v4)
        v6 = self.avgpool2(v5)
        v7 = self.conv4(v6)
        return v7
# Inputs to the model
x1 = torch.randn(4, 3, 64, 64)
