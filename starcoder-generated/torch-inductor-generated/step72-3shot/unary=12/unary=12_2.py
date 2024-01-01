
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 32, 1, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.conv3 = torch.nn.Conv2d(32, 32, 1, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(32, 32, 1, stride=1, padding=1)
        self.conv5 = torch.nn.Conv2d(32, 32, 1, stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(32)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.bn1(v2)
        v4 = self.conv3(v3)
        v5 = self.conv4(v4)
        v6 = self.conv5(v5)
        v7 = self.bn2(v6)
        v8 = F.sigmoid(v7)
        v9 = v7 * v8
        return v9
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
