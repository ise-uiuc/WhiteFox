
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(23, 14, 3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(14)
        self.conv2 = torch.nn.Conv2d(14, 5, 1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(10, 7, 1, stride=1, padding=0)
        self.conv4 = torch.nn.Conv2d(4, 1, 1, stride=1, padding=0)
        self.bn2 = torch.nn.BatchNorm2d(6)
        self.bn3 = torch.nn.BatchNorm2d(4)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.bn1(v1)
        v3 = torch.relu(v2)
        v4 = self.conv2(v3)
        v5 = self.conv3(v3)
        v6 = self.conv4(v3)
        v7 = torch.cat([v4, v5, v6], 1)
        v8 = torch.tanh(v7)
        v9 = torch.relu(v8)
        v10 = self.bn3(v9)
        v11 = self.bn2(v10)
        return v11
# Inputs to the model
x1 = torch.randn(1, 23, 56, 56)
