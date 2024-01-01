
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 7, stride=1, padding=3, groups=1, bias=True, dilation=1)
        self.conv2 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1, groups=1, bias=True, dilation=1)
        self.conv3 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1, groups=1, bias=True, dilation=1)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = torch.clamp_min(v3, self.min)
        v5 = torch.clamp_max(v4, self.max)
        return v5
min = -4.43272961502e-308
max = 4.43272961502e-308
# Inputs to the model
x1 = torch.randn(1, 3, 28, 28)
