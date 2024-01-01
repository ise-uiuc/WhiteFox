
class Model(torch.nn.Module):
    def __init__(self, kernel_size, padding, min, max):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, kernel_size, stride=1, padding=padding)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, self.min)
        v3 = torch.clamp_max(v2, self.max)
        return v3
kernel_size = 3
padding = 0
min = 2.0000000372529023
max = 0.199999994947575
# Inputs to the model
x1 = torch.randn(2, 1, 38, 38)
