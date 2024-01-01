
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
        self.conv1 = torch.nn.Conv2d(3, 4, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 3, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(4, 1, 3, stride=1, padding=1)
        self.bn = torch.nn.BatchNorm2d(4)
    def forward(self, x1):
        v1 = self.conv0(x1)
        v2 = self.conv1(x1)
        v3 = self.conv2(x1)
        v4 = v1 + v2
        v5 = v3 + 3.5
        v6 = torch.clamp_min(v4, 0.5)
        v7 = torch.clamp_max(v5, 6)
        v8 = torch.relu(v6)
        v9 = torch.relu6(v7)
        v10 = torch.relu6(v9)
        v11 = v1 * v8
        v12 = v10 / 6
        v13 = self.conv3(x1)
        v14 = v13 + v8
        x2 = v12 + v14
        v15 = self.bn(x2)
        return v15
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
