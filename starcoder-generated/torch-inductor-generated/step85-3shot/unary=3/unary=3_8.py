
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(512, 8, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(8, 2, 1, stride=1, padding=0)
        self.conv4 = torch.nn.Conv2d(2, 2, 1, stride=1, padding=0)
        self.conv5 = torch.nn.Conv2d(2, 15, 1, stride=1, padding=0)
        self.conv6 = torch.nn.Conv2d(15, 35, 1, stride=1, padding=0)
        self.conv7 = torch.nn.Conv2d(35, 10, 1, stride=1, padding=0)
        self.conv8 = torch.nn.Conv2d(10, 2, 3, stride=1, padding=1)
        self.conv9 = torch.nn.Conv2d(2, 20, 1, stride=1, padding=0)
        self.conv10 = torch.nn.Conv2d(20, 9, 1, stride=1, padding=0)
        self.conv11 = torch.nn.Conv2d(9, 9, 3, stride=1, padding=1)
        self.conv12 = torch.nn.Conv2d(9, 13, 3, stride=1, padding=1)
        self.conv13 = torch.nn.Conv2d(13, 13, 1, stride=1, padding=0)
        self.conv14 = torch.nn.Conv2d(13, 5, 1, stride=1, padding=0)
        self.conv15 = torch.nn.Conv2d(5, 36, 1, stride=1, padding=0)
        self.conv16 = torch.nn.Conv2d(36, 9, 1, stride=1, padding=0)
        self.conv17 = torch.nn.Conv2d(9, 6, 3, stride=1, padding=1)
        self.conv18 = torch.nn.Conv2d(6, 30, 1, stride=1, padding=0)
        self.conv19 = torch.nn.Conv2d(30, 9, 1, stride=1, padding=0)
        self.conv20 = torch.nn.Conv2d(9, 7, 3, stride=1, padding=1)
        self.conv21 = torch.nn.Conv2d(7, 7, 1, stride=1, padding=0)
        self.conv22 = torch.nn.Conv2d(7, 14, 1, stride=1, padding=0)
        self.conv23 = torch.nn.Conv2d(14, 17, 1, stride=1, padding=0)
        self.conv24 = torch.nn.Conv2d(17, 21, 3, stride=1, padding=1)
        self.conv25 = torch.nn.Conv2d(21, 15, 1, stride=1, padding=0)
        self.conv26 = torch.nn.Conv2d(15, 21, 1, stride=1, padding=0)
        self.conv27 = torch.nn.Conv2d(21, 8, 3, stride=1, padding=1)
        self.conv28 = torch.nn.Conv2d(8, 16, 1, stride=1, padding=0)
        self.conv29 = torch.nn.Conv2d(16, 13, 1, stride=1, padding=0)
        self.conv30 = torch.nn.Conv2d(13, 17, 1, stride=1, padding=0)
        self.conv31 = torch.nn.Conv2d(17, 22, 1, stride=1, padding=0)
        self.conv32 = torch.nn.Conv2d(22, 29, 1, stride=1, padding=0)
        self.conv33 = torch.nn.Conv2d(29, 32, 3, stride=1, padding=1)
        self.conv34 = torch.nn.Conv2d(32, 12, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        v7 = self.conv2(v6)
        v8 = v7 * 0.5
        v9 = v7 * 0.7071067811865476
        v10 = torch.erf(v9)
        v11 = v10 + 1
        v12 = v8 * v11
        v13 = self.conv3(v12)
        v14 = self.conv4(v13)
        v15 = self.conv5(v14)
        v16 = self.conv6(v15)
        v17 = self.conv7(v16)
        v18 = self.conv8(v17)
        v19 = self.conv9(v18)
        v20 = v19 * 0.5
        v21 = v19 * 0.7071067811865476
        v22 = torch.erf(v21)
        v23 = v22 + 1
        v24 = v20 * v23
        v25 = self.conv10(v24)
        v26 = self.conv11(v25)
        v27 = self.conv12(v26)
        v28 = self.conv13(v27)
        v29 = self.conv14(v28)
        v30 = self.conv15(v29)
        v31 = self.conv16(v30)
        v32 = self.conv17(v31)
        v33 = self.conv18(v32)
        v34 = self.conv19(v33)
        v35 = self.conv20(v34)
        v36 = self.conv21(v35)
        v37 = self.conv22(v36)
        v38 = self.conv23(v37)
        v39 = self.conv24(v38)
        v40 = self.conv25(v39)
        v41 = self.conv26(v40)
        v42 = self.conv27(v41)
        v43 = self.conv28(v42)
        v44 = self.conv29(v43)
        v45 = self.conv30(v44)
        v46 = self.conv31(v45)
        v47 = self.conv32(v46)
        v48 = self.conv33(v47)
        v49 = self.conv34(v48)
        return v49
# Inputs to the model
x1 = torch.randn(1, 512, 14, 14)
