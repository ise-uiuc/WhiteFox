
class Model(torch.nn.Module):
    def __init__(self, negative_slope = 0.25):
        super().__init__()
        self.conv = torch.nn.Conv2d(32, 1, (1, 3), 1, padding=(0, 1), groups=1, bias=False)
        self.conv_t = torch.nn.ConvTranspose2d(1, 32, 4, stride=3, padding=1, groups=1, dilation=1, output_padding=0, bias=False)
    def forward(self, img):
        x = self.conv(img)
        x = self.conv_t(x)
        y = x > 0
        z = torch.where(y, x, x * negative_slope)
        return z
negative_slope = 2.5
# Inputs to the Model
img = torch.randn(1, 32, 64, 64)
