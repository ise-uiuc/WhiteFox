
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_7 = torch.nn.ConvTranspose2d(126, 126, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_transpose_8 = torch.nn.ConvTranspose2d(126, 91, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_transpose_9 = torch.nn.ConvTranspose2d(91, 56, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_transpose_10 = torch.nn.ConvTranspose2d(56, 46, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_transpose_11 = torch.nn.ConvTranspose2d(46, 36, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_transpose_12 = torch.nn.ConvTranspose2d(36, 26, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_transpose_13 = torch.nn.ConvTranspose2d(26, 9, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    def forward(self, x1):
        v1 = self.conv_transpose_7(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v4 = self.conv_transpose_8(v3)
        v5 = torch.sigmoid(v4)
        v6 = v4 * v5
        v7 = self.conv_transpose_9(v6)
        v8 = torch.sigmoid(v7)
        v9 = v7 * v8
        v10 = self.conv_transpose_10(v9)
        v11 = torch.sigmoid(v10)
        v12 = v10 * v11
        v13 = self.conv_transpose_11(v12)
        v14 = torch.sigmoid(v13)
        v15 = v13 * v14
        v16 = self.conv_transpose_12(v15)
        v17 = torch.sigmoid(v16)
        v18 = v16 * v17
        v19 = self.conv_transpose_13(v18)
        return v19
# Inputs to the model
x1 = torch.randn(1, 126, 224, 224)
