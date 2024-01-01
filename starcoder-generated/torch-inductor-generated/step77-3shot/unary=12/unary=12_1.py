
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.conv2 = torch.nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(32)
        self.conv3 = torch.nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(32)
        self.conv4 = torch.nn.Conv2d(32, 1, 3, stride=1, padding=1)
        self.bn4 = torch.nn.BatchNorm2d(1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v4 = self.bn1(v1)
        v5 = self.conv2(x1)
        v6 = self.bn2(v5)
        v7 = self.conv3(x1)
        v8 = self.bn3(v7)
        v9 = v1 + v5
        v10 = (v4 - v6) * v9 - v9
        v11 = v10 + v7
        v12 = v11
        v13 = v10
        v14 = v12 * v13
        return v14
# Inputs to the model
x1 = torch.randn(3, 32, 224, 224)
