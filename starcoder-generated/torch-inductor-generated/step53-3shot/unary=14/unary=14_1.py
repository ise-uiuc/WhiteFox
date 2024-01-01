
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_17 = torch.nn.ConvTranspose2d(197, 512, 3, stride=1, padding=1, groups=64, bias=True)
        self.conv_transpose_19 = torch.nn.ConvTranspose2d(196, 512, 3, stride=1, padding=1, groups=64, bias=True)
        self.conv_transpose_21 = torch.nn.ConvTranspose2d(196, 512, 3, stride=1, padding=1, groups=64, bias=True)
        self.conv_transpose_23 = torch.nn.ConvTranspose2d(196, 511, 3, stride=1, padding=1, output_padding=1, groups=64, bias=True)
        self.conv_transpose_25 = torch.nn.ConvTranspose2d(195, 509, 3, stride=1, padding=1, output_padding=1, groups=64, bias=True)
        self.conv_transpose_27 = torch.nn.ConvTranspose2d(195, 508, 3, stride=1, padding=1, output_padding=0, groups=64, bias=True)
        self.conv_transpose_29 = torch.nn.ConvTranspose2d(195, 506, 3, stride=1, padding=1, groups=64, bias=True)
        self.conv_transpose_31 = torch.nn.ConvTranspose2d(194, 1968, 2, stride=2, padding=0, groups=64, bias=True)
    def forward(self, x1, x2, x3, x4, x5, x6, x7, x8):
        v1 = self.conv_transpose_17(x8)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v4 = self.conv_transpose_19(x5, v3)
        v5 = torch.sigmoid(v4)
        v6 = v4 * v5
        v7 = self.conv_transpose_21(v6)
        v8 = torch.sigmoid(v7)
        v9 = v7 * v8
        v10 = self.conv_transpose_23(x1, v9)
        v11 = torch.sigmoid(v10)
        v12 = v10 * v11
        v13 = self.conv_transpose_25(x2, v12)
        v14 = torch.sigmoid(v13)
        v15 = v13 * v14
        v16 = self.conv_transpose_27(v15)
        v17 = torch.sigmoid(v16)
        v18 = v16 * v17
        v19 = self.conv_transpose_29(x3, v18)
        v20 = torch.sigmoid(v19)
        v21 = v19 * v20
        v22 = self.conv_transpose_31(x4, v21)
        return v22
# Inputs to the model
x1 = torch.randn(1, 197, 10, 10)
x2 = torch.randn(1, 196, 33, 33)
x3 = torch.randn(1, 196, 67, 67)
x4 = torch.randn(1, 196, 100, 100)
x5 = torch.randn(1, 195, 100, 100)
x6 = torch.randn(1, 195, 100, 100)
x7 = torch.randn(1, 195, 100, 100)
x8 = torch.randn(1, 194, 202, 202)
