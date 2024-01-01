
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.conv3 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(8)
        self.conv5 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
        self.conv6 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(8)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(v1)
        v3 = self.bn1(v2)
        v4 = self.conv3(v3)
        v5 = self.conv4(v4)
        v6 = self.bn2(v5)
        v7 = self.conv5(v6)
        v8 = self.conv6(v7)
        v9 = self.bn3(v8)
        return v9

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 8, 1, stride=1, padding=1),
            nn.Conv2d(8, 8, 1, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 8, 1, stride=1, padding=1),
            nn.Conv2d(8, 8, 1, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 8, 1, stride=1, padding=1),
            nn.Conv2d(8, 8, 1, stride=1, padding=1),
            nn.BatchNorm2d(8),
        )
    def forward(self, x):
        return self.model(x)
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
