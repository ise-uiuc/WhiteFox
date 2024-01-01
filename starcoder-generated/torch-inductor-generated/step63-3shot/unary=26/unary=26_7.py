
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(80192, 7, (1, 1), stride=(1, 1), bias=False, dilation=(1, 1),
                                             groups=1, padding=(0, 0), output_padding=(0, 0),
                                             padding_mode='zeros')
    def forward(self, x7):
        x8 = self.conv_t(x7)
        x9 = x8 > 0
        x10 = x8 * -0.0152
        x11 = torch.where(x9, x8, x10)
        x12 = x11 + torch.nn.functional.adaptive_avg_pool2d(x11, (1, 1))
        return x12
# Inputs to the model
x7 = torch.randn(47, 80192, 1, 1)
