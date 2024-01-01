
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose3d(768, 768, 3, stride=1, padding=2, output_padding=1, bias=False, dilation=2, groups=2)
        self.negative_slope = negative_slope
    def forward(self, x2):
        r1 = self.conv_t(x2)
        r2 = r1 > 0
        r3 = r1 * -0.072369
        r4 = torch.where(r2, r1, r3)
        x6 = torch.neg(r4)
        x7 = torch.nn.functional.relu6(x6)
        return x7
negative_slope = -0.05
# Inputs to the model
x2 = torch.randn(5, 768, 135, 67, 29)
