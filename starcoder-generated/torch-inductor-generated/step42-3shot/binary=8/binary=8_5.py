
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.bn2 = torch.nn.BatchNorm2d(8)
        self.conv3 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(8)
        self.bn4 = torch.nn.BatchNorm2d(8)
        self.conv5 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv6 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.bn5 = torch.nn.BatchNorm2d(8)
        self.bn6 = torch.nn.BatchNorm2d(8)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = self.conv2(x2)
        v3 = v1 + v2
        v4 = self.bn1(v3)
        v5 = self.bn2(v3)
        v6 = self.conv3(x1)
        v7 = self.conv4(x2)
        v8 = v6 + v7
        v9 = self.bn3(v8)
        v10 = self.bn4(v8)
        v11 = v9 * v10
        v12 = self.conv5(x1)
        v16 = self.conv6(x2)
        v13 = v12 - v16
        v14 = self.bn5(v13)
        v17 = self.bn6(v13)
        v18 = v14.div(v17)
        return v11 * v18
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
