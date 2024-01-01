
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(4, 128, kernel_size=(1, 6), padding=(0, 3), dilation=3, output_padding=1, groups=2)
    def forward(self, x5):
        x1 = self.conv_t(x5)
        x2 = x1 > 0
        x3 = x1 * -4.94
        x4 = torch.where(x2, x1, x3)
        return x4
# Inputs to the model
x5 = torch.randn(1, 4, 35, 42)
