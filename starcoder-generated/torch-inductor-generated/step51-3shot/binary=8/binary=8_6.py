
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.bn2 = torch.nn.BatchNorm2d(8)
        self.conv3 = torch.nn.Conv2d(8, 32, 1, stride=2, padding=1)
        self.conv4 = torch.nn.Conv2d(8, 32, 1, stride=2, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(32)
        self.bn4 = torch.nn.BatchNorm2d(32)
        self.conv5 = torch.nn.Conv2d(32, 128, 1, stride=1, padding=1)
        self.conv6 = torch.nn.Conv2d(32, 128, 1, stride=1, padding=1)
        self.conv7 = torch.nn.Conv2d(32, 128, 3, stride=1, padding=1)
        self.conv8 = torch.nn.Conv2d(32, 128, 3, stride=1, padding=1)
        self.bn5 = torch.nn.BatchNorm2d(128)
        self.bn6 = torch.nn.BatchNorm2d(128)
        self.bn7 = torch.nn.BatchNorm2d(128)
        self.bn8 = torch.nn.BatchNorm2d(128)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = self.conv2(x2)
        v3 = self.bn1(v1)
        v4 = self.bn2(v2)
        v5 = self.conv3(v3)
        v6 = self.conv4(v4)
        v7 = self.bn3(v5)
        v8 = self.bn4(v6)
        v9 = self.conv5(v7)
        v10 = self.conv6(v7)
        v11 = self.conv7(v7)
        v12 = self.conv8(v7)
        v13 = self.bn5(v9)
        v14 = self.bn6(v10)
        v15 = self.bn7(v11)
        v16 = self.bn8(v12)
        v17 = v13.add(v15)
        v18 = self.bn5(v14)
        v19 = self.bn6(v16)
        v20 = self.bn7(v17)
        v21 = v18.mul(v19)
        v22 = v20.div(v21)
        return v22
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
