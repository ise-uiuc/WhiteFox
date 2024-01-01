
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_t = torch.nn.ConvTranspose2d(64, 128, 4, stride=2, padding=0, output_padding=0, groups=1, bias=True)
    def forward(self, x91):
        t1 = self.conv_t(x91)
        t2 = t1 > 0
        t3 = t1 * 0.88173
        t4 = torch.where(t2, t1, t3)
        return t4
# Inputs to the model
x91 = torch.randn(8, 64, 56, 56)
