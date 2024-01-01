
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 9, stride=2, padding=3, bias=False)
        self.bn = torch.nn.BatchNorm2d(3, track_running_stats=True)
        self.bn1 = torch.nn.BatchNorm2d(3, track_running_stats=False)
        self.bn2 = torch.nn.BatchNorm2d(3, track_running_stats=True)
        self.bn3 = torch.nn.BatchNorm2d(3, track_running_stats=False)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.relu(v1)
        v3 = self.bn(v2)
        v4 = torch.relu(v3)
        v5 = self.bn1(v4)
        v6 = torch.relu(v5)
        v7 = self.bn2(v6)
        v8 = torch.relu(v7)
        v9 = self.bn3(v8)
        v10 = torch.relu(v9)
        v11 = torch.sigmoid(v10)
        return v11
# Inputs to the model
x1 = torch.randn(1, 3, 41, 41)
