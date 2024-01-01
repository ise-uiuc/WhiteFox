
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 1, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(16)
        self.conv2 = torch.nn.Conv2d(16, 16, 1, stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(16)
        self.conv3 = torch.nn.Conv2d(16, 16, 1, stride=1, padding=3)
        self.bn3 = torch.nn.BatchNorm2d(16)
        self.conv4 = torch.nn.Conv2d(16, 16, 1, stride=1, padding=1)
        self.bn4 = torch.nn.BatchNorm2d(16)
        self.conv5 = torch.nn.Conv2d(16, 16, 1, stride=1, padding=3)
        self.bn5 = torch.nn.BatchNorm2d(16)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.bn1(v1)
        v3 = self.conv2(v2)
        v4 = self.bn2(v3)
        v5 = self.conv3(v4)
        v6 = self.bn3(v5)
        v7 = nn.functional.conv2d(v6, 33, 1, padding=3, bias=None, stride=1, dilation=3, groups=1)
        v8 = self.bn4(v7)
        v9 = self.conv4(v8)
        v10 = self.bn5(v9)
        v11 = nn.functional.conv2d(v10, 33, 1, padding=3, bias=None, stride=1, dilation=3, groups=1)
        v12 = torch.add(v6, v11, alpha=1)
        v13 = v12 + 0.1
        return v13
# Inputs to the model
x = torch.randn(1, 3, 32, 32)
