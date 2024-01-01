
class ConvBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(16)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(32)
        self.conv3 = torch.nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(64)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.nn.functional.relu(v1)
        v3 = self.bn1(v2)
        v4 = self.conv2(v3)
        v5 = torch.nn.functional.relu(v4)
        v6 = self.bn2(v5)
        v7 = self.conv3(v6)
        v8 = torch.nn.functional.relu(v7)
        v9 = self.bn3(v8)
        return v9
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer4 = ConvBlock()
        self.layer5 = ConvBlock()
    def forward(self, x1):
        v1 = self.layer4(x1)
        v2 = self.layer5(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
