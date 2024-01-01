
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3, stride=2, padding=1)
        self.pool = torch.nn.AvgPool2d(kernel_size=12)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.pool(v1) + 3
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp(v3, 0, 6)
        v5 = v1 * v4
        v6 = v5 / 6
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
