
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=3, padding=0, dilation=1, groups=1, bias=False)
        self.gelu = torch.nn.GELU()
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = self.gelu(v1)
        v4 = torch.nn.functional.interpolate(v2, scale_factor=(8.275799046688208,), recompute_scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor_for_mode='none')
        v5 = v4 + 3
        v6 = torch.clamp(v5, min=0)
        v7 = torch.clamp(v6, max=6)
        v8 = v2 * v7
        v9 = v8 / 6
        return v9
# Inputs to the model
x1 = torch.randn(1, 64, 64, 64)
