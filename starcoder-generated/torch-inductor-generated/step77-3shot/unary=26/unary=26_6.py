
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(13, 76, 4, stride=3, padding=0, dilation=1, groups=3, bias=True)
    def forward(self, x22):
        m1 = self.conv_t(x22)
        m2 = m1 > 0
        m3 = m1 * 8
        m4 = torch.where(m2, m1, m3)
        return m4
# Inputs to the model
x22 = torch.randn(3, 13, 68, 51)
