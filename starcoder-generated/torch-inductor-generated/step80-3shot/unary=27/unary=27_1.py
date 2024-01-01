
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv = torch.nn.Conv2d(62, 32, 1, stride=1, padding=0)
        self.avg_pool2d = torch.nn.AvgPool2d(kernel_size=6, stride=6, padding=5)
        self.conv1 = torch.nn.Conv2d(32, 31, 1, stride=1, padding=0)
        self.avg_pool2d_1 = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=1)
        self.conv_1 = torch.nn.Conv2d(31, 60, 2, stride=2, padding=0)
        self.avg_pool2d_2 = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=1)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.avg_pool2d(v1)
        v3 = self.conv1(v2)
        v4 = self.avg_pool2d_1(v3)
        v5 = self.conv_1(v4)
        v6 = self.avg_pool2d_2(v5)
        v7 = torch.clamp_min(v6, self.min)
        v8 = torch.clamp_max(v7, self.max)
        return v8


min = -0.1
max = 0.6
# Inputs to the model
x1 = torch.randn(1, 62, 8, 8)
