
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(1, 7, 5, stride=2, padding=3, groups=1, dilation=2, bias=True, padding_mode='zeros')
    def forward(self, x):
        x1 = self.conv_t(x)
        x2 = x1 > 0
        x3 = x1 * 3.7
        x4 = torch.where(x2, x1, x3)
        return x4
# Inputs to the model
x = torch.randn(7, 1, 7, 7)
