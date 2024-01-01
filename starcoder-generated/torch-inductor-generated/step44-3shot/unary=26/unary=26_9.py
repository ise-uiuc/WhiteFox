
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(10, 10, 3, bias=False, padding=(3, 0))
    def forward(self, x4):
        m1 = self.conv_t(x4)
        m2 = m1 > 0
        m3 = m1 * -0.30569046525001526
        m4 = torch.where(m2, m1, m3)
        return torch.nn.functional.conv2d(m4, weight=torch.ones(10, 25, 3, 2), bias=None, stride=(90, 90), padding=None, dilation=1, groups=1)
# Inputs to the model
x4 = torch.randn(27, 10, 88, 22)
