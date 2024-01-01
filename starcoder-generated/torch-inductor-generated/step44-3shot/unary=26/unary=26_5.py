
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(7, 8, 10, bias=False)
    def forward(self, x0):
        m1 = self.conv_t(x0)
        m2 = m1 > -0.5
        m3 = m1 * -0.1905
        m4 = torch.where(m2, m1, m3)
        return m4
# Inputs to the model
x0 = torch.randn(6, 7, 6, 9, device='cuda')
