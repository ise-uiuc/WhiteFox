
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(28, 27, 1, stride=1, padding=0)
    def forward(self, x):
        x1 = self.conv_t(x)
        x2 = x1 > 0
        x3 = x1 * 2.35
        x4 = torch.where(x2, x1, x3)
        x5 = torch.nn.functional.pad(x4, (11, 15, 11, 15, 0, 0, 0, 0), "constant", 0)
        x6 = torch.nn.functional.interpolate(x5, size=7, mode="bilinear", align_corners=False)
        return torch.nn.functional.interpolate(x6, scale_factor=0.603488, mode="nearest")
# Inputs to the model
x = torch.randn(1, 28, 97, 96)
