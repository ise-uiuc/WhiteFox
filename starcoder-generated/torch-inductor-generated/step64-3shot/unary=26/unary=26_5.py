
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(49, 50, 7, stride=2, padding=0)
    def forward(self, x1):
        f1 = self.conv_t(x1)
        f2 = f1 > 0
        f3 = f1 * 0.080
        f4 = torch.where(f2, f1, f3)
        return torch.nn.functional.interpolate(f4, size=14, mode='bilinear', align_corners=False)
# Inputs to the model
x1 = torch.randn(1, 49, 28, 17)
