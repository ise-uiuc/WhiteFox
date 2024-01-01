
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 5, stride=1, padding=2)
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.bn2 = torch.nn.BatchNorm2d(8)
        self.conv2 = torch.nn.Conv2d(8, 8, 5, stride=2, padding=2)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.bn1(v1)
        v3 = v2 - 0.1
        v4 = F.relu(v3)
        v5 = self.bn2(v4)
        v6 = v5 - 0.1
        v7 = F.relu(v6)
        v8 = self.conv2(v7)
        v9 = self.bn(v8)
        v10 = v9 - 0.1
        v11 = F.relu(v10)
        return v11
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
