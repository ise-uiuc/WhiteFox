
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose1d(2, 8, 4, stride=2, padding=1, output_padding=(3, 0), dilation=2, groups=1, bias=False)
        self.conv_transpose_1 = torch.nn.ConvTranspose1d(8, 16, 5, stride=2, padding=1, output_padding=3, dilation=2, groups=1, bias=True)
        self.conv_transpose_2 = torch.nn.ConvTranspose1d(16, 4, 4, stride=2, padding=2, output_padding=(2, 3), dilation=2, groups=1, bias=True)
        self.gelu = torch.nn.GELU()
    def forward(self, x1):
        x = x1
        x3 = self.conv_transpose(x)
        x4 = self.conv_transpose_1(x3)
        x5 = self.conv_transpose_2(x4)
        return self.gelu(x1)
# Inputs to the model
x1 = torch.randn(7, 2, 6)
