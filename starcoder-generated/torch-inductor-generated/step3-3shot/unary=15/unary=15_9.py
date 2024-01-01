
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.conv2 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(8)
        self.conv3 = torch.nn.Conv2d(8, 16, 5, stride=1, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(16)
    def forward(self, x1):
        v1 = self.bn1(self.conv1(x1))
        v2 = self.bn2(self.conv2(v1))
        v3 = self.bn3(self.conv3(v2))
        v4 = torch.relu(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
