
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 12, 5, stride=1, padding=2, bias=True)
        self.bn1 = torch.nn.BatchNorm2d(12)
        self.conv2 = torch.nn.Conv2d(12, 16, 3, stride=1, padding=1, bias=True)
        self.bn2 = torch.nn.BatchNorm2d(16)
        self.conv3 = torch.nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=True)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.bn1(v1)
        v3 = F.relu(v2)
        v4 = self.conv2(v3)
        v5 = self.bn2(v4)
        v6 = F.relu(v5)
        v7 = self.conv3(v6)
        v8 = v7 - 2
        v9 = F.relu(v8)
        return v9
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
