
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
        self.bn = torch.nn.BatchNorm2d(3)
        self.relu6 = torch.nn.ReLU6(inplace=True)
        self.conv1 = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(3)
        self.relu61 = torch.nn.ReLU6(inplace=True)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = 3 + v1
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 6)
        v5 = v1 * v4
        v6 = v5 / 6
        v7 = self.bn(v6)

        v8 = self.relu6(v7)
        v9 = self.conv1(v8)
        v10 = v9 + 3
        v11 = torch.clamp_min(v10, 0)
        v12 = torch.clamp_max(v11, 6)
        v13 = v9 * v12
        v14 = v13 / 6
        v15 = self.bn(v14)
        v16 = self.relu61(v15)

        return v16
# Inputs to the model
x1 = torch.randn(2, 3, 32, 32)
