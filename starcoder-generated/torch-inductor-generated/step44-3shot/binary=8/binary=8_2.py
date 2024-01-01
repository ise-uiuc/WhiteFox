
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 8, 1, stride=2, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.bn2 = torch.nn.BatchNorm2d(8)
        self.conv3 = torch.nn.Conv2d(3, 8, 1, stride=2, padding=1)
        self.conv4 = torch.nn.Conv2d(3, 8, 1, stride=2, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(8)
        self.bn4 = torch.nn.BatchNorm2d(8)
        self.conv5 = torch.nn.Conv2d(3, 8, 1, stride=2, padding=1)
        self.conv6 = torch.nn.Conv2d(3, 8, 1, stride=2, padding=1)
        self.bn5 = torch.nn.BatchNorm2d(8)
        self.bn6 = torch.nn.BatchNorm2d(8)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = self.conv2(x2)
        v3 = self.bn1(v1)
        v4 = self.bn2(v2)
        v5 = v3 + v4
        v6 = self.conv3(x1)
        v7 = self.conv4(x1)
        v8 = self.conv5(x2)
        v9 = self.conv6(x2)
        v10 = v6 + v7 + v8 - v9
        v11 = self.bn3(v10)
        v12 = self.bn4(v10)
        v13 = self.bn5(v10)
        v14 = self.bn6(v10)
        v15 = v11 - v12 + v13 + v14
        return v5.div(v15)
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
x2 = torch.randn(1, 3, 32, 32)
