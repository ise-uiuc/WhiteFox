
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, 7, stride=1, padding=0, bias=True)
        self.bn = torch.nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = torch.nn.ReLU()
        self.avg_pool2d = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = torch.flatten
        self.linear = torch.nn.Linear(16, 1)
        self.sigmoid = torch.sigmoid
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.bn(v1)
        v3 = self.relu(v2)
        v4 = self.avg_pool2d(v3)
        v5 = self.flatten(v4, 1)
        v6 = self.linear(v5)
        v7 = self.sigmoid(v6)
        v8 = torch.clamp_min(v7, self.min)
        v9 = torch.clamp_max(v8, self.max)
        return v9
min = -0.47
max = -0.25
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
