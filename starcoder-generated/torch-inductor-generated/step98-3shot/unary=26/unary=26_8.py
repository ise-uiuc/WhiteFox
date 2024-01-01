
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose3d(32, 32, 6, stride=8, padding=5, dilation=5, output_padding=2, bias=False)
    def forward(self, x14):
        x1 = self.conv_t(x14)
        x2 = x1 > 0
        x3 = x1 * -0.011
        x4 = torch.where(x2, x1, x3)
        return torch.nn.functional.adaptive_avg_pool3d(torch.nn.ReLU()(x4), (26, 38, 7))
# Inputs to the model
x14 = torch.randn(28, 32, 16, 19, 49)
