
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.bn2 = torch.nn.BatchNorm2d(8)
        self.bn3 = torch.nn.BatchNorm2d(8)
        self.conv4 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.bn2(v1)
        v3 = v2.mul(v2)
        v4 = self.bn1(v3)
        v5 = self.bn3(v3)
        v6 = v4 + v5
        v7 = torch.sin(v1)
        v8 = torch.sin(v1)
        v9 = v7.add(v8, alpha=1.0)
        v10 = self.conv3(v1)
        v11 = torch.sin(v1)
        v12 = v11.mul(v6)
        v13 = self.conv2(v1)
        v14 = torch.sin(v1)
        v15 = self.bn1(v9)
        v16 = v14.sub(v15)
        v17 = torch.sin(v1)
        v18 = v11.add(v15)
        v19 = v17.sub(v18)
        v20 = v12 * v16
        v21 = v20 * v19
        return v21
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
