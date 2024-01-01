
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Pointwise convolution 1
        self.conv_transpose_1 = torch.nn.ConvTranspose1d(64, 32, 5, stride=1, padding=0, output_padding=0, dilation=2, groups=1, bias=False)
        # Pointwise convolution 2
        self.conv_transpose_2 = torch.nn.ConvTranspose1d(32, 32, 5, stride=3, padding=1, output_padding=1, dilation=2, groups=2, bias=True)
        # Pointwise convolution 3
        self.conv_transpose_3 = torch.nn.ConvTranspose1d(32, 32, 5, stride=2, padding=1, output_padding=1, dilation=1, groups=8, bias=False)
        # Pointwise convolution 4
        self.conv_transpose_4 = torch.nn.ConvTranspose1d(32, 32, 5, stride=2, padding=8, output_padding=2, dilation=1, groups=20, bias=True)
    def forward(self, x1):
        # Pointwise convolution 1
        v1 = self.conv_transpose_1(x1)
        v2 = torch.tanh(v1)
        # Pointwise convolution 2
        v3 = self.conv_transpose_2(v2)
        v4 = torch.tanh(v3)
        # Pointwise convolution 3
        v5 = self.conv_transpose_3(v4)
        v6 = torch.tanh(v5)
        # Pointwise convolution 4
        v7 = self.conv_transpose_4(v6)
        v8 = torch.tanh(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 64, 17)
