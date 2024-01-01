
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 1, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(3, 25, (1, 7), stride=2, padding=(0, 3))
        self.conv3 = torch.nn.Conv2d(25, 31, (1, 5), stride=2, padding=1)
        self.conv4 = torch.nn.Conv2d(31, 26, (1, 5), stride=2, padding=1)
        self.conv5 = torch.nn.Conv2d(26, 28, (1, 5), stride=2, padding=1)
        self.conv6 = torch.nn.Conv2d(28, 23, (1, 5), stride=2, padding=1)
        self.conv7 = torch.nn.Conv2d(23, 24, (1, 5), stride=2, padding=1)
        self.conv8 = torch.nn.Conv2d(24, 20, (1, 5), stride=2, padding=1)
        self.conv9 = torch.nn.Conv2d(20, 19, (1, 5), stride=2, padding=1)
        self.conv10 = torch.nn.Conv2d(19, 17, (1, 5), stride=2, padding=1)
        self.conv11 = torch.nn.Conv2d(17, 14, (1, 5), stride=2, padding=1)
        self.conv12 = torch.nn.Conv2d(14, 10, (1, 5), stride=2, padding=1)
        self.conv13 = torch.nn.Conv2d(10, 9, 1, stride=1, padding=0)
        self.conv14 = torch.nn.Conv2d(10, 12, 1, stride=1, padding=0)
        self.conv15 = torch.nn.Conv2d(14, 8, 1, stride=1, padding=0)
        self.conv16 = torch.nn.Conv2d(17, 8, 1, stride=1, padding=0)
        self.conv17 = torch.nn.Conv2d(19, 10, 1, stride=1, padding=0)
        self.conv18 = torch.nn.Conv2d(20, 7, 1, stride=1, padding=0)
        self.conv19 = torch.nn.Conv2d(23, 8, 1, stride=1, padding=0)
        self.conv20 = torch.nn.Conv2d(24, 3, 1, stride=1, padding=0)
        self.conv21 = torch.nn.Conv2d(28, 10, 1, stride=1, padding=0)
        self.conv22 = torch.nn.Conv2d(26, 5, 3, stride=1, padding=1)
        self.conv23 = torch.nn.Conv2d(31, 5, 1, stride=1, padding=0)
        self.conv24 = torch.nn.Conv2d(25, 3, 7, stride=1, padding=3)
        self.conv25 = torch.nn.Conv2d(3, 11, (1, 7), stride=2, padding=(0, 3))
        self.conv26 = torch.nn.Conv2d(11, 12, (1, 7), stride=2, padding=(0, 3))
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
        v26 = v25 * 0.5
        v27 = v25 * 0.7071067811865476
        v28 = torch.erf(v27)
        v29 = v28 + 1
        v30 = v26 * v29
        v31 = self.conv11(v30)
        v32 = self.conv12(v31)
        v33 = self.conv13(v32)
        v34 = self.conv14(v33)
        v35 = self.conv15(v34)
        v36 = self.conv16(v35)
        v37 = self.conv17(v36)
        v38 = self.conv18(v37)
        v39 = self.conv19(v38)
        v40 = self.conv20(v39)
        v41 = v40 * 0.5
        v42 = v40 * 0.7071067811865476
        v43 = torch.erf(v42)
        v44 = v43 + 1
        v45 = v41 * v44
        v46 = self.conv21(v45)
        v47 = self.conv22(v46)
        v48 = self.conv23(v47)
        v49 = self.conv24(v48)
        v50 = self.conv25(v49)
        v51 = self.conv26(v50)
        return v51
# Inputs to the model
x1 = torch.randn(1, 3, 360, 640)
