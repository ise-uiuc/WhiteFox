
class Model(torch.nn.Module):
    def __init__(self, shape, out_ch1, out_ch2, out_ch3):
        super().__init__()
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(shape[0], out_ch1, 3, stride=1)
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(out_ch1, out_ch2, 3, stride=2)
        self.conv_transpose_3 = torch.nn.ConvTranspose2d(out_ch2, out_ch3, 4, stride=2)
        self.conv_transpose_4 = torch.nn.ConvTranspose2d(out_ch3, out_ch3, 7, stride=1)
    def forward(self, x1, x2, x3):
        v1 = self.conv_transpose_1(x1)
        v2 = self.conv_transpose_2(v1)
        v3 = self.conv_transpose_3(v2 + x3)
        v4 = self.conv_transpose_4(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 32, 16, 16)
x2 = torch.randn(1, 16, 32, 32)
x3 = torch.randn(1, 19, 64, 64)
