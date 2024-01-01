
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_t = torch.nn.ConvTranspose2d(68, 90, 3, stride=2, padding=32, bias=False)
    def forward(self, x15):
        d1 = self.conv_t(x15)
        d2 = d1 > 0
        d3 = d1 * -1.3364
        d4 = torch.where(d2, d1, d3)
        return d4
# Inputs to the model
x15 = torch.randn(32, 68, 17, 13)
