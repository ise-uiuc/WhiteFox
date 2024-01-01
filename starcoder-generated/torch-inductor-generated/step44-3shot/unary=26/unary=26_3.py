
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(21, 21, 31, stride=3, padding=19, groups=2, dilation=2, bias=False)
    def forward(self, x1):
        m1 = self.conv_t(x1)
        m2 = m1 > 0
        m3 = m1 * -0.010884604
        m4 = torch.where(m2, m1, m3)
        return torch.nn.functional.interpolate(m4, scale_factor=0.36386494, mode='bilinear', align_corners=False)
# Inputs to the model
x1 = torch.randn(13, 21, 32, 64, device='cuda')
