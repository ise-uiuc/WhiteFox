
class Model(torch.nn.Module):
    def __init__(self, min_value=5.7897051314283525, max_value=-4.310529656290872):
        super().__init__()
        self.max_pool2d = torch.nn.MaxPool2d(2)
        self.avg_pool2d = torch.nn.AvgPool2d(1)
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=2, padding=5, groups=2, dilation=2)
        self.sub = torch.nn.MaxUnpool2d(3, 1, 2)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1, x2):
        v1 = self.conv(x2)
        v2 = self.max_pool2d(v1)
        v3 = self.avg_pool2d(x1)
        v4 = self.sub(v2, v3, 1)
        v5 = self.max_pool2d(x2)
        v6 = torch.clamp_min(v4, self.min_value)
        v7 = torch.clamp_max(v6, self.max_value)
        v8 = self.avg_pool2d(v5)
        return torch.flatten(v8, 1)
# Inputs to the model
x1 = torch.randn(1, 96, 16, 16)
x2 = torch.randn(1, 24, 8, 8)
