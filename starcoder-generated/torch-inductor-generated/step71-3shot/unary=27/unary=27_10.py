
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.avg_pool2d = torch.nn.AvgPool2d(kernel_size=(1, 2), stride=(2, 2))
        self.conv = torch.nn.Conv2d(1, 3, 2, stride=2, padding=3)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.avg_pool2d(x1)
        v2 = self.conv(v1)
        v3 = torch.clamp_min(v2, self.min)
        v4 = torch.clamp_max(v3, self.max)
        return v4
min = 1e-06
max = 0
# Inputs to the model
x1 = torch.randn(1, 1, 3, 16)
