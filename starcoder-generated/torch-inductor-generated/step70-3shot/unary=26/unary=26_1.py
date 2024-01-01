
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(64, 64, 3, padding=1, bias=False)
    def forward(self, x7):
        m1 = self.conv_t(x7)
        m2 = m1 > 0
        m3 = m1 * 0.036
        m4 = torch.where(m2, m1, m3)
        return m4
# Inputs to the model
x7 = torch.randn(6, 64, 34, 31)
