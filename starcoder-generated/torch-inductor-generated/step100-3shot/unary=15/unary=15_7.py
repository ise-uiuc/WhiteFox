
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn1 = torch.nn.BatchNorm2d(64, 96)
        self.bn2 = torch.nn.BatchNorm2d(64, 96)
        self.bn3 = torch.nn.BatchNorm2d(64, 96)
        self.conv1 = torch.nn.Conv2d(64, 64, (1,96), (1,96), (0,0))
        self.conv2 = torch.nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 64, 1, stride=1, padding=0)
        self.conv4 = torch.nn.Conv2d(64, 64, 5, stride=1, padding=2)
        self.conv5 = torch.nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv6 = torch.nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv7 = torch.nn.Conv2d(64, 64, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.bn1(x1)
        v4 = self.bn2(v1)
        v7 = self.bn3(v4)
        v8 = self.conv1(v7)
        v2 = self.conv2(v8)
        v3 = self.conv3(v2)
        v4 = self.conv4(v3)
        v5 = self.conv5(v4)
        v6 = self.conv6(v5)
        v3 = self.conv7(v6)
        v4 = torch.add(v3, v7)
        return v4
# Inputs to the model
x1 = torch.randn(1, 64, 32, 96)
