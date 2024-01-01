
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(12, 41, kernel_size=16, stride=(1, 1), padding=0, output_padding=0, dilation=1, groups=1, bias=True)
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(12, 72, kernel_size=1, stride=(1, 1), padding=0, output_padding=0, dilation=1, groups=1, bias=True)
        self.conv_transpose_3 = torch.nn.ConvTranspose2d(12, 48, kernel_size=8, stride=(4, 4), padding=0, output_padding=0, dilation=1, groups=1, bias=True)
        self.conv_transpose_4 = torch.nn.ConvTranspose2d(12, 48, kernel_size=7, stride=(3, 3), padding=0, output_padding=0, dilation=1, groups=1, bias=True)
        self.conv_transpose_5 = torch.nn.ConvTranspose2d(12, 24, kernel_size=11, stride=(5, 5), padding=0, output_padding=0, dilation=1, groups=1, bias=True)
        self.conv_transpose_6 = torch.nn.ConvTranspose2d(12, 62, kernel_size=5, stride=(3, 3), padding=0, output_padding=0, dilation=1, groups=1, bias=True)
        self.conv_transpose_7 = torch.nn.ConvTranspose2d(21, 53, kernel_size=7, stride=(8, 8), padding=2, output_padding=0, dilation=1, groups=1, bias=True)
    def forward(self, x1):
        v1 = self.conv_transpose_1(x1)
        v2 = torch.sigmoid(v1)
        v3 = self.conv_transpose_2(x1)
        v4 = torch.sigmoid(v3)
        v5 = self.conv_transpose_3(x1)
        v6 = torch.sigmoid(v5)
        v7 = self.conv_transpose_4(x1)
        v8 = torch.sigmoid(v7)
        v9 = self.conv_transpose_5(x1)
        v10 = torch.sigmoid(v9)
        v11 = self.conv_transpose_6(x1)
        v12 = torch.sigmoid(v11)
        v13 = self.conv_transpose_7(x1)
        v14 = torch.sigmoid(v13)
        v15 = v1 + v2
        v16 = v15 + v4
        v17 = v16 + v6
        v18 = v17 + v8
        v19 = v18 + v10
        v20 = v19 + v12
        v21 = v20 + v14
        return v21        
# Inputs to the model
x1 = torch.randn(1, 12, 28, 21)
