
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(655, 16, 3, stride=1, padding=0, dilation=1, groups=1)
        self.conv_transpose_3 = torch.nn.ConvTranspose2d(16, 1, 1, stride=1, padding=0, output_padding=0, groups=1)
    def forward(self, x1):
        v1 = self.conv_transpose_2(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v4 = self.conv_transpose_3(v3)
        v5 = torch.sigmoid(v4)
        v6 = v4 * v5
        return v6
x1 = torch.randn(1, 655, 48, 48)
