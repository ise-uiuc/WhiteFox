
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(64, 64, 3, stride=1, padding=1, groups=2)
        self.conv2 = torch.nn.Conv2d(64, 64, 3, stride=2, padding=1, dilation=1)
        self.conv3 = torch.nn.Conv2d(64, 64, 3, stride=1, padding=1, groups=2)
        self.conv5 = torch.nn.Conv2d(64, 64, 3, stride=1, padding=1, dilation=1)
        self.conv5 = torch.nn.Conv2d(64, 128, 1, stride=1, padding=0)
        self.conv6 = torch.nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv7 = torch.nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = v3
        v5 = self.conv5(v4)
        v6 = v5 + v3
        v7 = self.conv6(v6)
        v7 = self.conv7(v7)
        v8 = torch.clamp_min(v7, self.min)
        v9 = torch.clamp_max(v8, self.max)
        return v9
min = 0.98
max = 0.23
# Inputs to the model
x1 = torch.randn(1, 64, 224, 224)
