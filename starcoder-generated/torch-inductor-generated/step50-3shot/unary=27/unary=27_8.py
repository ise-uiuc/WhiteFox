
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 3, stride=2, padding=1, dilation=1, groups=1, bias=False)
        self.pointwise_conv = torch.nn.Conv2d(32, 32, 1, stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.relu = torch.nn.ReLU6(inplace=True)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, stride=2, padding=1, dilation=1, groups=1, bias=False)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.pointwise_conv(v1)
        v3 = self.relu(v2)
        v4 = self.conv2(v3)
        v5 = torch.clamp_min(v4, self.min)
        v6 = torch.clamp_max(v5, self.max)
        return v6
min = -1.7014118346046448e+38
max = 1.7014118346046448e+38
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
