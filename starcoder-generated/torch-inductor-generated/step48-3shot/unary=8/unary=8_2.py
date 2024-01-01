
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(384, 512, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.nn.functional.interpolate(v1, scale_factor=2, mode='bilinear', align_corners=False)
        v3 = v2 + 3
        v4 = torch.clamp(v3, min=0)
        v5 = torch.clamp(v4, max=6)
        v6 = v2 * v5
        v7 = v6 / 6
        return v7
# Inputs to the model
x1 = torch.randn(1, 384, 27, 27)
