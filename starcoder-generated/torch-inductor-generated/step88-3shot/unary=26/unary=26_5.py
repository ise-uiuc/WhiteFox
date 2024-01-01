
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(6, 9, 6, stride=2, padding=3, bias=False)
    def forward(self, x2):
        x1 = self.conv_t(x2)
        x2 = x1 > 0
        x3 = x1 * 0.012
        x4 = torch.where(x2, x1, x3)
        return torch.nn.functional.interpolate(x4, scale_factor=2.1, mode='bilinear', align_corners=False)
# Inputs to the model
x2 = torch.randn(1, 6, 224, 322)
