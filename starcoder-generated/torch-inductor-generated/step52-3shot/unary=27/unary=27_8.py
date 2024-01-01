
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv5 = torch.nn.Conv2d(256, 128, 3, stride=1, padding=1)
        self.conv6 = torch.nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.conv7 = torch.nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.conv8 = torch.nn.Conv2d(32, 3, 3, stride=1, padding=1)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = self.conv4(v3)
        v5 = self.conv5(v4)
        v6 = self.conv6(v5)
        v7 = self.conv7(v6)
        v8 = self.conv8(v7)
        v8 = torch.clamp_min(v8, self.min)
        v8 = torch.clamp_max(v8, self.max)
        return v8
min = 0.0012
max = 0.0906
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
