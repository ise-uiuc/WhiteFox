
class Model(torch.nn.Module):
    def __init__(self, min, max, kernel_size):
        super(Model, self).__init__()
        self.conv = torch.nn.Conv2d(1, 2, kernel_size, stride=2)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, self.min)
        v3 = torch.clamp_max(v2, self.max)
        return v3

min = -1.0
max = 1.0
kernel_size = 3
# Inputs to the model
x1 = torch.randn(1, 1, 9, 15)
