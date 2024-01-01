
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(42, 40, 7, stride=1, padding=0, bias=False)
    def forward(self, x4):
        m1 = self.conv_t(x4)
        m2 = m1 > 0
        m3 = m1 * -0.0041451187
        m4 = torch.where(m2, m1, m3)
        return torch.nn.functional.interpolate(m4, scale_factor=2, mode='bicubic', align_corners=False)
# Inputs to the model
x4 = torch.randn(5, 42, 86, 53)
