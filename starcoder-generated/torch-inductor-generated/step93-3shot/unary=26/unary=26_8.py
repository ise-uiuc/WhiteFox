
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dconv_t = torch.nn.ConvTranspose2d(128, 128, 1, 1, bias=True)
        self.conv_t = torch.nn.ConvTranspose2d(128, 32, 2, 1, bias=False)
    def forward(self, x):
        b = -self.dconv_t(x)
        c = self.conv_t(b)
        d = c > 0
        e = c * 0
        f = torch.where(d, c, e)
        return -torch.sum(f)
# Inputs to the model
x = torch.randn(1, 128, 30, 24)
