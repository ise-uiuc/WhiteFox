
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_t = torch.nn.ConvTranspose2d(3, 3, 2, stride=2, padding=1, output_padding=1, dilation=1, groups=1, bias=False, padding_mode='zeros')
    def forward(self, x6):
        t1 = self.conv_t(x6)
        t2 = t1 > 0
        t3 = t1 * -0.62298
        t4 = torch.where(t2, t1, t3)
        return t4
# Inputs to the model
x6 = torch.randn(64, 3, 18, 18)
