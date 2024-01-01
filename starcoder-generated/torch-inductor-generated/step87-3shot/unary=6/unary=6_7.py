
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 1, 1, stride=1, padding=0)
        self.bn1 = torch.nn.BatchNorm2d(3)
        self.bn2 = torch.nn.BatchNorm2d(3)
        self.bn3 = torch.nn.BatchNorm2d(1)
        self.relu = torch.nn.ReLU(inplace=True)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.bn1(3 + v1)
        v3 = 3 + v2
        v4 = torch.clamp_min(v3, 0)
        v5 = torch.clamp_max(v4, 6)
        v6 = v1 * v5
        v7 = v6 / 6
        v8 = self.conv2(v7)
        v9 = self.bn2(3 + v8)
        v10 = 3 + v9
        v11 = torch.clamp_min(v10, 0)
        v12 = torch.clamp_max(v11, 6)
        v13 = v8 * v12
        v14 = v13 / 6
        v15 = self.bn3(v14)
        v16 = torch.clamp_min(v15, 0)
        v17 = torch.clamp_max(v16, 6)
        v18 = v14 * v17
        v19 = v18 / 6
        return v19
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
