
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 8, 1, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(3, 8, 1, stride=2, padding=1)
        self.conv4 = torch.nn.Conv2d(3, 8, 1, stride=2, padding=1)
        self.conv5 = torch.nn.Conv2d(3, 8, 1, stride=2, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.bn2 = torch.nn.BatchNorm2d(8)
        self.bn3 = torch.nn.BatchNorm2d(8)
        self.bn4 = torch.nn.BatchNorm2d(8)
        self.bn5 = torch.nn.BatchNorm2d(8)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = self.conv2(x2)
        v3 = v1 + v2
        v4 = self.conv3(x1)
        v5 = self.conv4(x1)
        v6 = self.conv5(x2)
        v7 = v4 + v5 + v6
        v8 = self.bn1(v3)
        v9 = self.bn2(v7)
        v10 = v3 + v8
        v11 = v7 + v8
        v12 = self.bn3(v10)
        v13 = self.bn4(v11)
        v14 = v12 + v13
        v15 = self.bn5(v10)
        return (v15, v14)
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
x2 = torch.randn(1, 3, 32, 32)
