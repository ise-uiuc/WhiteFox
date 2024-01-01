
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.conv3 = torch.nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(32)
        self.conv4 = torch.nn.Conv2d(32, 16, 3, stride=1, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(16)
        self.conv5 = torch.nn.Conv2d(16, 1, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.bn1(v2)
        v4 = torch.relu(v3)
        v5 = self.conv3(v4)
        v6 = self.bn2(v5)
        v7 = torch.relu(v6)
        v8 = self.conv4(v7)
        v9 = self.bn3(v8)
        v10 = torch.relu(v9)
        v11 = self.conv5(v10)
        return v11
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
