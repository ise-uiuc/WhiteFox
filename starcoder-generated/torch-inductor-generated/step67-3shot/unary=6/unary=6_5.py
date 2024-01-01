
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
        self.relu6 = torch.nn.ReLU6(inplace=True)
        self.conv2 = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
        self.bn = torch.nn.BatchNorm2d(3)
        self.relu6 = torch.nn.ReLU6(inplace=True)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 + 3
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 6)
        v5 = v1 * v4
        v6 = v5 / 6
        v7 = self.relu6(v6)
        v8 = self.conv2(v7)
        v9 = v8 + 3
        v10 = torch.clamp_min(v9, 0)
        v11 = torch.clamp_max(v10, 6)
        v12 = v8 * v11
        v13 = v12 / 6
        v14 = self.relu6(v13)
        v15 = self.bn(v14)
        return v15.unsqueeze(-1).unsqueeze(-1)
# Inputs to the model
x1 = torch.randn(2, 3, 32, 32)
