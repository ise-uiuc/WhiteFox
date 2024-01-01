
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(128, 2, 7, stride=3, padding=1, output_padding=0)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = v1 + 3
        v3 = torch.clamp(v2, min=0)
        v4 = torch.clamp(v3, max=6)
        v5 = v1 * v4
        v6 = v5 / 6
        return torch.nn.functional.interpolate(v6, size=None, scale_factor=None, mode='bilinear', align_corners=False)
# Inputs to the model
x1 = torch.randn(1, 128, 24, 24)
