
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 26, 2, stride=1, padding=1, groups=2, dilation=3)
        self.adaptive_avg_pool2d = torch.nn.AdaptiveAvgPool2d(56)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = v1 + 3
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 6)
        v5 = v4 / 6
        v6 = self.adaptive_avg_pool2d(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 33, 77)
