
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, 1, stride=1, padding=0, dilation=1)
        self.pointwise_conv = torch.nn.Conv2d(16, 3, 1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1.sigmoid()
        v3 = v1 * v2
        v4 = self.pointwise_conv(v1)
        v5 = v4.clamp(min=0, max=256)
        v6 = v3 * v5
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
