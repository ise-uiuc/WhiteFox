
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(2048, 512, 1, stride=1, padding=0, bias=False)
    def forward(self, x41):
        x42 = self.conv_t(x41)
        x43 = x42 > 0
        x44 = x42 * 0.373
        x45 = torch.where(x43, x42, x44)
        x46 = torch.nn.functional.interpolate(x45, size=x14.shape[-2:], mode="nearest", align_corners=None)
        x47 = x41 * x2
        x48 = torch.nn.functional.adaptive_avg_pool2d(x41, (1, 1))
        x49 = torch.cat([(x47 + x48), x46], 1)
        x50 = self.conv_0(x49)
        x51 = x50 + torch.nn.functional.interpolate(x50, scale_factor=1.0007, mode="nearest", align_corners=None)
        return x28
# Inputs to the model
x41 = torch.randn(1, 2048, 32, 32)
x14 = torch.randn(1, 512, 32, 32)
x2 = torch.randn(1, 2048, 1, 1)
