
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 3, stride=1, padding=2, groups=1, bias=False, dilation=1)
        self.conv2 = torch.nn.Conv2d(6, 6, 3, stride=1, padding=1, groups=1, bias=False, dilation=1)
        self.add = torch.nn.quantized.FloatFunctional()
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.add.add_relu(v1, v2)
        v4 = torch.clamp_min(v3, self.min)
        v5 = torch.clamp_max(v4, self.max)
        return v5
min = 1.7014118346046442e-38
max = 2
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
