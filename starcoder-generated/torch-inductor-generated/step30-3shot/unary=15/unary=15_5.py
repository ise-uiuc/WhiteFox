
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 128, 3, stride=1, padding=0, dilation=1)
        self.bn1 = torch.nn.BatchNorm2d(128)
        self.conv2 = torch.nn.Conv2d(128, 64, 3, stride=1, padding=0, dilation=1)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.conv3 = torch.nn.Conv2d(64, 128, 1, stride=1, padding=0, dilation=1)
        self.bn3 = torch.nn.BatchNorm2d(128)
    def forward(self, x1):
        v1 = self.bn1(self.conv1(x1))
        v2 = torch.relu(v1)
        v3 = self.bn2(self.conv2(v2))
        v4 = torch.relu(v3)
        v5 = self.bn3(self.conv3(v4))
        v6 = torch.relu(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
