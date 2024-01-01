
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(4, 128, 7, stride=2, padding=3, dilation=3, output_padding=2, groups=2)
    def forward(self, x3):
        x1 = self.conv_t(x3)
        x2 = x1 > 0
        x3 = x1 * -4.94
        x4 = torch.where(x2, x1, x3)
        return x4
# Inputs to the model
x3 = torch.randn(1, 4, 35, 42)
