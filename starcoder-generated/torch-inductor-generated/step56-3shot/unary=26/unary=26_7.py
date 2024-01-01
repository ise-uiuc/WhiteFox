
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(77, 14, 9, stride=2, padding=4, dilation=2, groups=3, bias=False)
    def forward(self, x2):
        x13 = self.conv_t(x2)
        x14 = x13 > 0.163
        x15 = x13 * -0.016
        x16 = torch.where(x14, x13, x15)
        return torch.nn.functional.interpolate(x16, size=116, mode='bilinear', align_corners=False)
# Inputs to the model
x2 = torch.randn(20, 77, 111, 25, device='cuda')
