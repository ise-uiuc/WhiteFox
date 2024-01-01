

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 5, stride=1, padding=2)
        self.conv2 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(8, 8, 3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.bn2 = torch.nn.BatchNorm2d(8)
        self.bn3 = torch.nn.BatchNorm2d(8)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = self.bn1(v2)
        v5 = v4+v3
        v6 = self.bn1(v5)
        v7 = torch.transpose(v6, 1, 2)
        return v7
x = torch.randn(1, 3, 48, 128)
