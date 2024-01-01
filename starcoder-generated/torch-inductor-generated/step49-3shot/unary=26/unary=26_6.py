
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_t = torch.nn.ConvTranspose2d(11, 430, 6, stride=11, padding=3, output_padding=6, dilation=2, bias=False)
    def forward(self, x10):
        t1 = self.conv_t(x10)
        t2 = t1 > 0
        t3 = t1 * -1.46
        t4 = torch.where(t2, t1, t3)
        return t4
# Inputs to the model
x10 = torch.randn(1, 11, 30, 53)
