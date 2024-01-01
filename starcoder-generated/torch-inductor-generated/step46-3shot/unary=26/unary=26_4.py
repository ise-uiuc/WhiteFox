
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(48, 22, 4, stride=[1, 1], padding=[1, 1], bias=False)
    def forward(self, x0):
        m1 = self.conv_t(x0)
        m2 = m1 > 0
        m3 = m1 * -0.072
        m4 = torch.where(m2, m1, m3)
        return m4
# Inputs to the model
x0 = torch.randn(50, 48, 23, 80)
