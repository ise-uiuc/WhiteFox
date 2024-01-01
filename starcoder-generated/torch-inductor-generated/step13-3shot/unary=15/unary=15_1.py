
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, (3, 3), stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(16)
        self.conv2 = torch.nn.Conv2d(16, 16, 1, stride=1, padding=0)
        self.bn2 = torch.nn.BatchNorm2d(16)
        self.conv3 = torch.nn.Conv2d(16, 32, (3, 3), stride=1, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(32)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.bn1(v1)
        v3 = torch.relu(v2)
        v4 = self.conv2(v3)
        v5 = self.bn2(v4)
        v6 = torch.relu(v5)
        v7 = self.conv3(v6)
        v8 = self.bn3(v7)
        v9 = torch.relu(v8)
        return v9
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
