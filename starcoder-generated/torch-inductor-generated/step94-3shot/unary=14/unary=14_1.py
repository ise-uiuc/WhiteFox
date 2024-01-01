
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_4 = torch.nn.ConvTranspose2d(32768, 65536, 7, stride=3, padding=3, output_padding=1, dilation=1, groups=32)
    def forward(self, x1):
        v1 = self.conv_transpose_4(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 32768, 44, 44)
