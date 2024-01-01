
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(8, 3, 3, stride=1, padding=1, dilation=1, groups=1, output_padding=0)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(10, 3, 3, stride=1, padding=1, dilation=1, groups=1, output_padding=0)
    def forward(self, x1):
        x2 = self.conv_transpose1(x1)
        x3 = self.conv_transpose2(torch.cat([x1, x2], 1))
        x4 = x3 + 3
        x5 = torch.clamp(x4, min=0)
        x6 = torch.clamp(x5, max=6)
        x7 = x3 * x6
        x8 = x7 / 6
        return x8
# Inputs to the model
x1 = torch.randn(1, 8, 64, 64)
