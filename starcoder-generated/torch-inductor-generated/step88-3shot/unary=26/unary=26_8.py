
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(3, 16, 7, stride=1, padding=1, bias=False)
    def forward(self, x):
        x0 = self.conv_t(x)
        x1 = x0 > 0
        x2 = x0 * -1.3153
        x3 = torch.where(x1, x0, x2)
        x4 = x3
        x5 = self.conv_t(x4)
        x6 = x5 > 0
        x7 = x5 * -6.5811
        x8 = torch.where(x6, x5, x7)
        x9 = x8
        x10 = self.conv_t(x9)
        x11 = x10 > 0
        x12 = x10 * -4.1687
        x13 = torch.where(x11, x10, x12)
        x14 = x13
        x15 = self.conv_t(x14)
        x16 = x15 > 0
        x17 = x15 * 0.0658
        x18 = torch.where(x16, x15, x17)
        x19 = x18
        return torch.nn.functional.interpolate(x19, size=29, mode='bilinear', align_corners=False)
# Inputs to the model
x = torch.randn(1, 3, 30, 13)
