
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(3)
        self.conv1 = torch.nn.Conv2d(3, 8, 5, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.conv2 = torch.nn.Conv2d(8, 8, 3, stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(8)
        self.conv3 = torch.nn.Conv2d(8, 16, 2, stride=1, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(16)
    def forward(self, x1):
        v1 = self.bn(x1)
        v2 = self.bn1(self.conv1(v1))
        v3 = self.bn2(self.conv2(v2))
        v4 = torch.relu(v3)
        v5 = self.bn3(self.conv3(v4))
        v6 = torch.relu(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 128, 128)
