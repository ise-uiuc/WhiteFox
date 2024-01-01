
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(1, 1, 6, stride=2, padding=2)
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(1, 2, 6, stride=2, padding=2)
        self.conv_transpose_3 = torch.nn.ConvTranspose2d(2, 3, 6, stride=2, padding=2)
        self.conv_transpose_4 = torch.nn.ConvTranspose2d(3, 4, 6, stride=2, padding=2)
        self.conv_transpose_5 = torch.nn.ConvTranspose2d(4, 5, 6, stride=2, padding=2)
        self.conv_transpose_6 = torch.nn.ConvTranspose2d(5, 6, 6, stride=2, padding=2)
        self.conv_transpose_7 = torch.nn.ConvTranspose2d(6, 7, 6, stride=2, padding=2)
        self.conv_transpose_8 = torch.nn.ConvTranspose2d(7, 8, 6, stride=2, padding=2)
        self.conv_transpose_9 = torch.nn.ConvTranspose2d(8, 9, 6, stride=2, padding=2)
        self.conv_transpose_10 = torch.nn.ConvTranspose2d(9, 10, 7, stride=1, padding=3)
        self.conv_transpose_11 = torch.nn.ConvTranspose2d(10, 11, 7, stride=1, padding=3)
        self.conv_transpose_12 = torch.nn.ConvTranspose2d(11, 12, 5, stride=1, padding=1, dilation=1)
        self.conv_transpose_13 = torch.nn.ConvTranspose2d(12, 13, 6, stride=1, padding=2, dilation=2)
        self.conv_transpose_14 = torch.nn.ConvTranspose2d(13, 14, 5, stride=1, padding=1, dilation=1)
        self.conv_transpose_15 = torch.nn.ConvTranspose2d(14, 15, 6, stride=1, padding=2, dilation=2)
        self.conv_transpose_16 = torch.nn.ConvTranspose2d(15, 16, 7, stride=1, padding=3, dilation=3)
        self.conv_transpose_17 = torch.nn.ConvTranspose2d(16, 17, 7, stride=1, padding=3, dilation=3)
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
        v26 = torch.sigmoid(v25)
        v27 = v25 * v26
        v28 = self.conv_transpose_10(v27)
        v29 = torch.sigmoid(v28)
        v30 = v28 * v29
        v31 = self.conv_transpose_11(v30)
        v32 = torch.sigmoid(v31)
        v33 = v31 * v32
        v34 = self.conv_transpose_12(v33)
        v35 = torch.sigmoid(v34)
        v36 = v33 + v35
        v37 = self.conv_transpose_13(v36)
        v38 = torch.sigmoid(v37)
        v39 = v37 + v38
        v40 = self.conv_transpose_14(v39)
        v41 = torch.sigmoid(v40)
        v42 = v39 + v41
        v43 = self.conv_transpose_15(v42)
        v44 = torch.sigmoid(v43)
        v45 = v43 + v44
        v46 = self.conv_transpose_16(v45)
        v47 = torch.sigmoid(v46)
        v48 = v45 + v47
        v49 = self.conv_transpose_17(v48)
        v50 = torch.sigmoid(v49)
        v51 = v49 * v50
        return v51
# Inputs to the model
x1 = torch.randn(1, 1, 28, 28)
