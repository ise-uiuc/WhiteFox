
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose1d(183, 181, 5, bias=False)
    def forward(self, x5):
        s1 = self.conv_t(x5)
        s2 = s1 > 0
        s3 = s1 * -0.207
        s4 = torch.where(s2, s1, s3)
        return s4
# Inputs to the model
x5 = torch.randn(215, 183, 52)
