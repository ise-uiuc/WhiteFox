
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_16 = torch.nn.ConvTranspose3d(16, 16, 5, stride=1, padding=1, output_padding=0, dilation=1)
        self.conv_transpose_18 = torch.nn.ConvTranspose3d(16, 16, 5, stride=2, padding=2, output_padding=0, dilation=1)
        self.conv_transpose_20 = torch.nn.ConvTranspose3d(16, 16, 5, stride=2, padding=2, output_padding=(1, 0, 0), dilation=1)
        self.conv_transpose_22 = torch.nn.ConvTranspose3d(16, 16, 5, stride=2, padding=2, output_padding=(1, 1, 1), dilation=1)
    def forward(self, x2):
        v1 = self.conv_transpose_16(x2)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v4 = self.conv_transpose_18(v3)
        v5 = torch.sigmoid(v4)
        v6 = v4 * v5
        v7 = self.conv_transpose_20(v6)
        v8 = torch.sigmoid(v7)
        v9 = v7 * v8
        v10 = self.conv_transpose_22(v9)
        return v10
# Inputs to the model
x2 = torch.randn(1, 16, 14, 14, 14)
