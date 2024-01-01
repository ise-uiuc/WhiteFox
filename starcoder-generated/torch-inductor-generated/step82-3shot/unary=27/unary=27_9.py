
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(in_channels=1, out_channels=2, kernel_size=2, stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.conv2d(x1)
        v2 = torch.clamp_min(v1, self.min)
        v3 = torch.clamp_max(v2, self.max)
        return v3
min = 1
max = 1.0
# Inputs to the model
x1 = torch.randn(1, 1, 3, 3)
