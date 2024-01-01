
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_t = torch.nn.ConvTranspose2d(3, 64, 2, stride=1, dilation=1, output_padding=1, groups=3, bias=True)
    def forward(self, x):
        x1 = self.conv_t(x)
        x2 = x1 > 0
        x3 = x1 * 0.056149
        x4 = torch.where(x2, x1, x3)
        return x4
# Inputs to the model
x = torch.randn(3, 3, 224, 224)
