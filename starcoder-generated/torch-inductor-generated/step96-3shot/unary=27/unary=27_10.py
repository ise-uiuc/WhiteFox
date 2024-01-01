
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.bn = torch.nn.BatchNorm2d(73, momentum=0.5)
        self.conv2 = torch.nn.Conv2d(55, 63, 3, stride=1, padding=0)
        self.relu = torch.nn.ReLU(True)
        self.max = max
        self.min = min
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.bn(v1)
        v3 = self.conv2(v2)
        v4 = torch.clamp_min(v3, self.min)
        v5 = torch.clamp_max(v4, self.max)
        v6 = self.relu(v5)
        return v6
min = -3.0
max = 3.0
# Inputs to the model
x = torch.randn(1, 32, 32, 64)
