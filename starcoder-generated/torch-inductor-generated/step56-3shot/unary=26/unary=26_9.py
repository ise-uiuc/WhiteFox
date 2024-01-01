
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(39, 42, 7, stride=2, padding=0)
    def forward(self, x5):
        m1 = self.conv_t(x5)
        m2 = m1 > 0
        m3 = m1 * 4.121
        m4 = torch.where(m2, m1, m3)
        return torch.nn.functional.interpolate(m4, size=4, mode='bicubic', align_corners=False)
# Inputs to the model
x5 = torch.randn(9, 39, 10, 12, dtype=torch.float, device='cuda')
