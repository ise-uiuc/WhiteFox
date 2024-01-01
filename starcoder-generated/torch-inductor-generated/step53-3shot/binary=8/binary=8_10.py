
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.bn2 = torch.nn.BatchNorm2d(8)
        self.conv3 = torch.nn.Conv2d(3, 16, 1, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(3, 16, 1, stride=1, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(16)
        self.bn4 = torch.nn.BatchNorm2d(16)
        self.conv5 = torch.nn.Conv2d(3, 32, 1, stride=1, padding=1)
        self.conv6 = torch.nn.Conv2d(3, 32, 1, stride=1, padding=1)
        self.bn5 = torch.nn.BatchNorm2d(32)
        self.bn6 = torch.nn.BatchNorm2d(32)
    def forward(self, x1, x2):
        v1 = self.bn1(self.conv1(x1))
        v2 = self.bn2(self.conv2(x2))
        v3 = v1 + v2
        v4 = v3 + self.bn3(self.conv3(x1))
        v5 = v4 * self.bn4(self.conv4(x2))
        v6 = v5 + self.bn5(self.conv5(x1))
        v7 = v6 + self.bn6(self.conv6(x2))
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
