
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_t = torch.nn.ConvTranspose2d(90, 39, 8, stride=4, padding=2, bias=True)
    def forward(self, x10):
        m1 = self.conv_t(x10)
        m2 = m1 > 0
        m3 = m1 * -0.101
        m4 = torch.where(m2, m1, m3)
        return torch.nn.functional.pad(m4, (4, 3, 2, 33))
# Inputs to the model
x10 = torch.randn(8, 90, 16, 84)
