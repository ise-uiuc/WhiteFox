
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(4096, 2048, 7, stride=1, padding=3)
        self.conv_transpose_3 = torch.nn.ConvTranspose2d(2048, 1638, 7, stride=1, padding=1)
        self.conv_transpose_4 = torch.nn.ConvTranspose2d(1638, 1154, 7, stride=2, padding=4)
        self.conv_transpose_5 = torch.nn.ConvTranspose2d(1154, 1024, 7, stride=1, padding=3)
        self.conv_transpose_6 = torch.nn.ConvTranspose2d(1024, 1024, 7, stride=1, padding=3)
        self.conv_transpose_7 = torch.nn.ConvTranspose2d(1024, 728, 7, stride=1, padding=3)
        self.conv_transpose_8 = torch.nn.ConvTranspose2d(728, 192, 7, stride=2, padding=4)
        self.conv_transpose_9 = torch.nn.ConvTranspose2d(192, 144, 7, stride=1, padding=3)
        self.conv_transpose_10 = torch.nn.ConvTranspose2d(144, 86, 7, stride=2, padding=2)
        self.conv_transpose_11 = torch.nn.ConvTranspose2d(86, 35, 7, stride=1, padding=3)
    def forward(self, x1):
        v1 = self.conv_transpose_2(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v4 = self.conv_transpose_3(v3)
        v5 = torch.sigmoid(v4)
        v6 = v4 * v5
        v7 = self.conv_transpose_4(v6)
        v8 = torch.sigmoid(v7)
        v9 = v7 * v8
        v10 = self.conv_transpose_5(v9)
        v11 = torch.sigmoid(v10)
        v12 = v10 * v11
        v13 = self.conv_transpose_6(v12)
        v14 = torch.sigmoid(v13)
        v15 = v13 * v14
        v16 = self.conv_transpose_7(v15)
        v17 = torch.sigmoid(v16)
        v18 = v16 * v17
        v19 = self.conv_transpose_8(v18)
        v20 = torch.sigmoid(v19)
        v21 = v19 * v20
        v22 = self.conv_transpose_9(v21)
        v23 = torch.sigmoid(v22)
        v24 = v22 * v23
        v25 = self.conv_transpose_10(v24)
        v26 = torch.sigmoid(v25)
        v27 = v25 * v26
        v28 = self.conv_transpose_11(v27)
        v29 = torch.sigmoid(v28)
        v30 = v28 * v29
        return v30
# Inputs to the model
x1 = torch.randn(1, 4096, 32, 32)
