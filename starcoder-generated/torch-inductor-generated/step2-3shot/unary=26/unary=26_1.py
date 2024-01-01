
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(3, 8, 1, stride=1, padding=1, groups=2, dilation=1, output_padding=1, bias=True)
    def forward(self, images, negative_slope):
        v1 = self.conv(images)
        v2 = v1 > 0
        v3 = v1 * negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
images = torch.randn(1, 15, 64, 64)
negative_slope = 0.5
