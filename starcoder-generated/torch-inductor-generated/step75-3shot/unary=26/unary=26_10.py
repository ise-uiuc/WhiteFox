
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(5, 128, 7, stride=2, padding=3, dilation=3, output_padding=1, groups=5)
    def forward(self, x49):
        o5 = self.conv_t(x49)
        o6 = o5 > 0
        o7 = o5 * -4.94
        o8 = torch.where(o6, o5, o7)
        return o8
# Inputs to the model
x49 = torch.randn(3, 5, 35, 42)
