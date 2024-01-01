
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.bn2 = torch.nn.BatchNorm2d(8)
        self.conv3 = torch.nn.Conv2d(8, 32, 1, stride=2, padding=1)
        self.conv4 = torch.nn.Conv2d(8, 32, 1, stride=2, padding=1)
        self.conv5 = torch.nn.Conv2d(32, 64, 1, stride=1, padding=1)
        self.conv6 = torch.nn.Conv2d(32, 128, 1, stride=1, padding=1)
        self.conv7 = torch.nn.Conv2d(64, 64, 1, stride=1, padding=1)
        self.conv8 = torch.nn.Conv2d(32, 16, 1, stride=1, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(32)
        self.bn4 = torch.nn.BatchNorm2d(32)
        self.bn5 = torch.nn.BatchNorm1d(64)
        self.bn6 = torch.nn.BatchNorm1d(128)
        self.bn7 = torch.nn.BatchNorm1d(64)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = self.conv2(x2)
        v3 = self.bn1(v1)
        v4 = self.bn2(v2)
        v5 = v3.add(v4)
        v6 = self.conv3(v5)
        v7 = self.conv4(v5)
        v8 = self.bn3(v6)
        v9 = self.bn4(v7)
        v10 = v8 + v9
        v11 = self.conv5(v10)
        v12 = self.conv6(v10)
        v13 = self.bn5(v11)
        v14 = self.bn6(v12)
        v15 = v13 - v14
        v16 = v13.mul(v14)
        v17 = v15 + v16
        v18 = self.conv7(v17)
        v19 = self.conv8(v10)
        v20 = self.bn7(v18)
        v21 = v19 - v20
        return v21.div(v20)
# Inputs to the model
x1 = torch.randn(1, 3, 27, 37)
x2 = torch.randn(1, 3, 27, 37)
