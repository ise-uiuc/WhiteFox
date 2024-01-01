
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 24, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(24, 32, 4, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(32, 64, 2, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(24)
        self.bn2 = torch.nn.BatchNorm2d(32)
        self.bn3 = torch.nn.BatchNorm2d(64)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.bn1(v1)
        v3 = F.relu(v2)
        v4 = self.conv2(v3)
        v5 = self.bn2(v4)
        v6 = F.relu(v5)
        v7 = self.conv3(v6)
        v8 = self.bn3(v7)
        v9 = F.relu(v8)
        v10 = self.conv4(v9)
        v11 = v10 - 0.5
        v12 = self.sigmoid(v11)
        v13 = torch.squeeze(v12, 0)
        return v13
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
