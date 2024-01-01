
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(1, 1, 8, stride=1, bias=True)
        self.conv = torch.nn.Conv2d(1, 1, 6, stride=2, padding=3, output_padding=1, groups=1, dilation=1, bias=False, padding_mode='zeros')
    def forward(self, x1, x2, x3, x4):
        x5 = self.conv_t(x4)
        x6 = x5 > 0
        x7 = x5 * -0.26
        x8 = torch.where(x6, x5, x7)
        x9 = self.conv(x8)
        return torch.relu6(x9)
# Inputs to the model
x1 = torch.randn(7, 1, 40, 10, device='cpu')
x2 = torch.randn(5, 1, 28, 15, device='cpu')
x3 = torch.randn(4, 1, 6, 4, device='cpu')
x4 = torch.randn(4, 1, 2, 9, device='cpu')
