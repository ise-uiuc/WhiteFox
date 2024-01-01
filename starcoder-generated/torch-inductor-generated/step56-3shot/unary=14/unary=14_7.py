
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_0 = torch.nn.ConvTranspose2d(1, 24, 3, stride=2, padding=0, dilation=1)
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(1, 32, 3, stride=2, padding=0, dilation=1)
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(16, 16, 1, stride=1, padding=0, dilation=1)
        self.conv_transpose_3 = torch.nn.ConvTranspose2d(9, 9, 1, stride=1, padding=0, dilation=1)
        self.conv_transpose_4 = torch.nn.ConvTranspose2d(96, 9, 1, stride=1, padding=0, dilation=1)
        self.conv_transpose_5 = torch.nn.ConvTranspose2d(32, 32, 5, stride=1, padding=0, dilation=1)
        self.conv_transpose_6 = torch.nn.ConvTranspose2d(48, 48, 1, stride=1, padding=0, dilation=1)
        self.conv_transpose_7 = torch.nn.ConvTranspose2d(8, 8, 1, stride=1, padding=0, dilation=1)
        self.conv_transpose_8 = torch.nn.ConvTranspose2d(16, 16, 1, stride=1, padding=0, dilation=1)
        self.conv_transpose_9 = torch.nn.ConvTranspose2d(16, 16, 1, stride=1, padding=0, dilation=1)
        self.conv_transpose_10 = torch.nn.ConvTranspose2d(160, 160, 1, stride=1, padding=0, dilation=1)
        self.conv_transpose_11 = torch.nn.ConvTranspose2d(48, 48, 3, stride=1, padding=0, dilation=1)
        self.conv_transpose_12 = torch.nn.ConvTranspose2d(48, 48, 1, stride=1, padding=0, dilation=1)
        self.conv_transpose_13 = torch.nn.ConvTranspose2d(64, 64, 1, stride=1, padding=0, dilation=1)
    def forward(self, x1):
        v1 = self.conv_transpose_0(x1)
        v2 = self.conv_transpose_1(v1)
        v3 = self.conv_transpose_2(v1)
        v4 = self.conv_transpose_3(v3)
        v5 = self.conv_transpose_4(v1)
        v6 = self.conv_transpose_5(v2)
        v7 = self.conv_transpose_6(v2)
        v8 = self.conv_transpose_7(v1)
        v9 = self.conv_transpose_8(v2)
        v10 = self.conv_transpose_9(v1)
        v11 = self.conv_transpose_10(v2)
        v12 = self.conv_transpose_11(v2)
        v13 = self.conv_transpose_12(v11)
        v14 = self.conv_transpose_13(v7)
        v15 = torch.sigmoid(v3)
        v16 = v14 * v15
        v17 = torch.tanh(v13)
        v18 = torch.sigmoid(v7)
        v19 = v14 * v15
        v20 = torch.tanh(v3)
        v21 = v7 * v8
        v22 = v21 * v20
        v23 = v22 + v5
        v24 = torch.tanh(v4)
        v25 = torch.sigmoid(v4)
        v26 = v17 + v18
        v27 = v19 * v20
        v28 = v16 + v23
        v29 = v6 + v27
        v30 = v10 * v20
        v31 = torch.tanh(v10)
        v32 = v10 * v11
        v33 = v19 * v32
        v34 = v33 + v8
        v35 = v34 + v23
        v3 = v7 * v8
        t1 = torch.sigmoid(v4)
        t2 = v16 + v24
        t3 = v16 + v25
        t4 = v28 + v29
        t5 = v28 + v35
        t6 = v30 + v31
        t7 = v30 + v35
        t8 = t2 + t6
        t9 = t3 + t7
        v40 = t5 * v40
        v37 = v4 - v40
        v38 = v13 * v38
        v36 = v16 + v17
        v39 = v36 + v37
        v40 = t8 + t9
        v2 = v38 + v39
        v40 = t1 + v40
        return v40
# Inputs to the model
x1 = torch.randn(1, 1, 1, 1)
