
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose2d = torch.nn.ConvTranspose2d(1, 1, kernel_size=3, stride=1, padding=1, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros')
    def forward(self, x5):
        x6 = self.conv_transpose2d(x5)
        x7 = x6 > 0
        x8 = x6 * 0.1
        x9 = torch.where(x7, x6, x8)
        return x9
# Inputs to the model
x5 = torch.randn(4, 1, 7, 7)
