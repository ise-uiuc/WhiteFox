
class Model(torch.nn.Module):
    def __init__(self, min_value, max_value, kernel_size):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, kernel_size, stride=2, padding=0,groups=3)
        self.conv2 = torch.nn.Conv2d(8, 256, 1, stride=1, padding=0, dilation=1, groups=1)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        v4 = self.conv2(v3).sum()
        return v4
min_value=0.02
max_value=3
kernel_size=7
# Inputs to the model
x1 = torch.randn(1, 3, 16, 16)
