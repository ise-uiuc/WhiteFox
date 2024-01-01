
class Model(torch.nn.Module):
    def __init__(self, min, max, stride, padding):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 2, stride1 stride, padding1 padding)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, self.min)
        v3 = torch.clamp_max(v2, self.max)
        return v3
min = 1
max = 2
stride = 1
padding = 1
# Inputs to the model
x1 = torch.randn(1, 3, 10, 10)
