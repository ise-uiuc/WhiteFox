
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(213, 205, 7, stride=4, padding=0, bias=False)
    def forward(self, x2):
        m1 = self.conv_t(x2)
        m2 = m1 > 0
        m3 = m1 * -0.88965902
        m4 = torch.where(m2, m1, m3)
        return m4
# Inputs to the model
x2 = torch.randn(2, 213, 21, 10)
