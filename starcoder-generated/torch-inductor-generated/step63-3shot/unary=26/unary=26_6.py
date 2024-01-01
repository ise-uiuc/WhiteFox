
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_t = torch.nn.ConvTranspose2d(3, 5, 3, stride=1, dilation=1, output_padding=1, padding=1, groups=1, bias=True)
    def forward(self, x34):
        t1 = self.conv_t(x34)
        t2 = t1 > 0
        t3 = t1 * 0.48
        t4 = torch.where(t2, t1, t3)
        return t4
# Inputs to the model
x34 = torch.randn(16, 3, 224, 224)
