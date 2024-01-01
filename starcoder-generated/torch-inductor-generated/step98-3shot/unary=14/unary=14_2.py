
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(17, 14, 7, stride=1, padding=3, dilation=1)
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(14, 8, 7, stride=1, padding=3)
        self.conv_transpose_3 = torch.nn.ConvTranspose2d(8, 8, 1, stride=1, padding=0, dilation=1)
        self.conv_transpose_4 = torch.nn.ConvTranspose2d(8, 5, 5, stride=1, padding=1)
        self.conv_transpose_5 = torch.nn.ConvTranspose2d(5, 5, 5, stride=1, padding=1)
        self.conv_transpose_6 = torch.nn.ConvTranspose2d(5, 5, 3, stride=1, padding=1)
        self.conv_transpose_7 = torch.nn.ConvTranspose2d(5, 6, 7, stride=1, padding=3)
        self.conv_transpose_8 = torch.nn.ConvTranspose2d(6, 5, 5, stride=1, padding=2)
        self.conv_transpose_9 = torch.nn.ConvTranspose2d(5, 3, 6, stride=1, padding=1)
    def forward(self, x1):
        f2 = torch.reshape(x1, (-1, 1, 6, 6))
        f4 = torch.reshape(x1, (-1, 1, 4, 6))
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
        v26 = torch.sigmoid(v25)
        v27 = v25 * v26
        o1 = torch.reshape(v27, (-1, 3, 6, 6))
        o2 = torch.reshape(o1, (-1, 3, 12, 6))
        o3 = torch.reshape(o2, (-1, 3, 36, 1))
        return o3
# Inputs to the model
x1 = torch.randn(1, 17, 18, 18)
