
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(160, 64, 2, stride=2, padding=(0, 0))
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(64, 128, 2, stride=2, padding=(1, 1))
        self.conv_transpose_3 = torch.nn.ConvTranspose2d(128, 16, 2, stride=2, padding=(0, 0))
        self.conv_transpose_4 = torch.nn.ConvTranspose2d(16, 8, 2, stride=2, padding=(1, 1))
        self.conv_transpose_5 = torch.nn.ConvTranspose2d(8, 8, 2, stride=2, padding=(0, 0))
        self.conv_transpose_6 = torch.nn.ConvTranspose2d(8, 8, 2, stride=2, padding=(1, 1))
        self.conv_transpose_7 = torch.nn.ConvTranspose2d(8, 64, 2, stride=2, padding=(0, 0))
        self.conv_transpose_8 = torch.nn.ConvTranspose2d(64, 512, 2, stride=2, padding=(1, 1))
        self.conv_transpose_9 = torch.nn.ConvTranspose2d(2048, 2, 2, stride=2, padding=(0, 1))
    def forward(self, x1):
        v1 = self.conv_transpose_1(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v4 = self.conv_transpose_2(v3)
        v5 = torch.sigmoid(v4)
        v6 = v4 * v5
        v7 = self.conv_transpose_3(v6)
        v8 = torch.sigmoid(v7)
        v9 = v7 * v8
        v10 = self.conv_transpose_4(v9)
        v11 = torch.sigmoid(v10)
        v12 = v10 * v11
        v13 = self.conv_transpose_5(v12)
        v14 = torch.sigmoid(v13)
        v15 = v13 * v14
        v16 = self.conv_transpose_6(v15)
        v17 = torch.sigmoid(v16)
        v18 = v16 * v17
        v19 = self.conv_transpose_7(v18)
        v20 = torch.sigmoid(v19)
        v21 = v19 * v20
        v22 = self.conv_transpose_8(v21)
        v23 = torch.sigmoid(v22)
        v24 = v22 * v23
        v25 = self.conv_transpose_9(v24)
        return v25
# Inputs to the model
x1 = torch.randn(1, 160, 32, 32)
