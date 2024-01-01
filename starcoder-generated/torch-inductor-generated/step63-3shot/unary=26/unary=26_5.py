
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(1, 1, 4, padding=3, groups=2, stride=2, dilation=1, bias=True, padding_mode='zeros')
    def forward(self, x6):
        y6 = self.conv_t(x6)
        y7 = y6 * 0.871
        y8 = torch.cat([x6, y7], 1)
        return y8
# Inputs to the model
x6 = torch.randn(1, 1, 15, 20)
