
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, 3, stride=2, padding=1, dilation=1, groups=1, bias=False)
        self.maxpool = torch.nn.MaxPool2d(2, stride=2)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, stride=2, padding=1, dilation=1, groups=1, bias=False)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.maxpool(v1)
        v3 = self.conv2(v2)
        v4 = torch.clamp_min(v3, self.min)
        v5 = torch.clamp_max(v4, self.max)
        return v5
min = 100
max = 100
# Inputs to the model
x1 = torch.randn(1, 1, 100, 100)
