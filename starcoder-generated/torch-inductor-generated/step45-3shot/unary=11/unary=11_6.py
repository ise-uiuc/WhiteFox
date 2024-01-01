
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(8, 16, 3)
    def forward(self, x1):
        t0 = torch.conv2d(input=x1, weight=0, bias=None, stride=1, padding=1, dilation=1, groups=1)
        v1 = self.conv_transpose(t0)
        v2 = v1 + 3
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 6)
        v5 = v4 / 6
        return v5
# Inputs to the model
x1 = torch.randn(1, 8, 64, 64)
