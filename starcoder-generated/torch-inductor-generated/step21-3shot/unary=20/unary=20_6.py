
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, out_channels=5, kernel_size=1, stride=1, padding=1, dilation=2, groups=1, bias=True, padding_mode='reflect')
        self.pixel_shuffle = torch.nn.PixelUnshuffle(upscale_factor=2)
    def forward(self, x1, x2):
        v1 = self.conv(x1)
        v2 = self.pixel_shuffle(v1)
        return v1, v2
# Inputs to the model
x1 = torch.randn(1, 3, 47, 112)
x2 = torch.randn(1, 5, 147, 224)
