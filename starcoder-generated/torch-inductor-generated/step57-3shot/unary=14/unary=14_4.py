
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_128_128_1 = torch.nn.ConvTranspose2d(128, 128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), output_padding=(0, 0), groups=1, bias=False, dilation=1)
        self.conv_transpose_128_28_1_group_16 = torch.nn.ConvTranspose2d(128, 28, (1, 1), stride=(1, 1), padding=(0, 0), groups=16, bias=False)
    def forward(self, x1, x2):
        v1 = self.conv_transpose_128_128_1(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v4 = self.conv_transpose_128_28_1_group_16(x1)
        return (v3, v4)
# Inputs to the model
input1 = torch.randn(1, 128, 64, 64)
input2 = torch.randn(1, 128, 16, 16)
