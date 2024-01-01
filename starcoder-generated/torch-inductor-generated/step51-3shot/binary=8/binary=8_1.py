
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 64, 1, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.conv3 = torch.nn.Conv2d(32, 32, 1, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(32, 32, 1, stride=1, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(32)
        self.bn4 = torch.nn.BatchNorm2d(32)
    def forward(self, x1, x2):
        v1 = self.bn1(self.conv1(x1))
        v2 = self.bn2(self.conv2(x2))
        v3 = self.conv3(v1)
        v4 = self.conv4(v2)
        v5 = self.bn3(v3)
        v6 = self.bn4(v4)
        v7 = v5 + v6
        v8 = self.bn3(torch.tanh(v5))
        v9 = self.bn4(torch.tanh(v6))
        v10 = v8 + v9
        v11 = self.bn3(torch.sigmoid(v5))
        v12 = self.bn4(torch.sigmoid(v6))
        v13 = v11 + v12
        return v10 * v13
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
