
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 9, 5, stride=_stride, padding=_padding, dilation=_dilation, groups=1, bias=False)
    def forward(self, x):
        negative_slope = 3.121348
        v1 = self.conv(x)
        v2 = v1 > 0
        v3 = v1 * negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x1 = torch.randn(28, 3, 63, 43)
_stride = 1
_padding = 1
_dilation = 1
