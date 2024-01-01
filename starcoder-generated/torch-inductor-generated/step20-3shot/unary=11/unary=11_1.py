
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(16, 32, [16, 1], stride=(4, 2), padding=(3, 0), dilation=(4, 1), groups=2, bias=False)
        self.bn = torch.nn.BatchNorm2d(32, 0.278783, 0.200667)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = v1 + 3
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 6)
        v5 = v4 / 6
        v6 = self.bn(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
