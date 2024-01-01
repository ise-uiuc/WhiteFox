
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 16, 1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(3, 32, 1, stride=1, padding=0)
        self.conv4 = torch.nn.Conv2d(3, 64, 1, stride=1, padding=0)
        self.conv5 = torch.nn.Conv2d(3, 128, 1, stride=1, padding=0)
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.bn2 = torch.nn.BatchNorm2d(16)
        self.bn3 = torch.nn.BatchNorm2d(32)
        self.bn4 = torch.nn.BatchNorm2d(64)
        self.bn5 = torch.nn.BatchNorm2d(128)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = self.conv2(x2)
        v3 = self.conv3(x1)
        v4 = self.conv4(x2)
        v5 = self.conv5(x1)

        v6 = self.bn1(v1)
        v7 = self.bn2(v2)
        v8 = self.bn3(v3)
        v9 = self.bn4(v4)
        v10 = self.bn5(v5)

        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
x2 = torch.randn(1, 3, 32, 32)
