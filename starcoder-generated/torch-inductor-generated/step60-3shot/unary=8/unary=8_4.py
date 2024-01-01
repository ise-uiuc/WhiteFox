
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(64, 16, 3, stride=1, dilation=1, padding=1, output_padding=0, groups=3, bias=True)
        self.conv_transpose_2d = torch.nn.ConvTranspose2d(64, 16, 3, stride=1, dilation=1, padding=1, output_padding=0, groups=3, bias=True)
        self.conv = torch.nn.Conv2d(16, 8, 7, stride=1, padding=3, dilation=2, groups=2)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = self.conv_transpose_2d(x1)
        v3 = v1 + v2 + 3
        v4 = torch.clamp(v3, min=0)
        v5 = torch.clamp(v4, max=6)
        v6 = v5 / 6
        v7 = self.conv(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 64, 3, 3)
