
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(39, 13, 6, padding=2, bias=False)
    def forward(self, x6):
        m1 = self.conv_t(x6)
        m2 = m1 * -0.033476262
        m3 = torch.where(m1 > 0, m2, m1)
        return m3
# Inputs to the model
x6 = torch.randn(8, 39, 70, 45)
