
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
        self.conv5 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
        self.conv6 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
        self.conv7 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
        self.conv8 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
        self.conv9 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
        self.conv10 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
        self.conv11 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
        self.conv12 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
        self.conv13 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
        self.conv14 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
        self.conv15 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
        self.conv16 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
        self.conv17 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
        self.conv18 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
        self.conv19 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
        self.conv20 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
        self.conv21 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
        self.conv22 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
        self.conv23 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
        self.conv24 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(x)
        v3 = v1 + v2
        v4 = self.conv3(v3)
        v5 = self.conv4(v3)
        v6 = v4 + v5
        v7 = self.conv5(v6)
        v8 = self.conv6(v6)
        v9 = v7 + v8
        v10 = self.conv7(v9)
        v11 = self.conv8(v9)
        v12 = v10 + v11
        v13 = self.conv9(v12)
        v14 = self.conv10(v12)
        v15 = v13 + v14
        x1 = self.conv11(v15)
        x2 = self.conv12(v15)
        x3 = x1 + x2
        x4 = self.conv13(x3)
        x5 = self.conv14(x3)
        x6 = x4 + x5
        x7 = self.conv15(x6)
        x8 = self.conv16(x6)
        x9 = x7 + x8
        x10 = self.conv17(x9)
        x11 = self.conv18(x9)
        x12 = x10 + x11
        x13 = self.conv19(x12)
        x14 = self.conv20(x12)
        x15 = x13 + x14
        x16 = self.conv21(x15)
        x17 = self.conv22(x15)
        x18 = x16 + x17
        x19 = self.conv23(x18)
        x20 = self.conv24(x18)
        x21 = x19 + x20
        x22 = x21.relu_()
        v23 = self.conv1(x1)
        v24 = self.conv2(x2)
        v25 = v23 + v24
        v26 = self.conv3(v25)
        v27 = self.conv4(v25)
        v28 = v26 + v27
        v29 = self.conv5(v28)
        v30 = self.conv6(v28)
        v31 = v29 + v30
        v32 = self.conv7(v31)
        v33 = self.conv8(v31)
        v34 = v32 + v33
        v35 = self.conv9(v34)
        v36 = self.conv10(v34)
        v37 = v35 + v36
        x1 = self.conv11(v37)
        x2 = self.conv12(v37)
        x3 = x1 + x2
        x4 = self.conv13(x3)
        x5 = self.conv14(x3)
        x6 = x4 + x5
        x7 = self.conv15(x6)
        x8 = self.conv16(x6)
        x9 = x7 + x8
        x10 = self.conv17(x9)
        x11 = self.conv18(x9)
        x12 = x10 + x11
        x13 = self.conv19(x12)
        x14 = self.conv20(x12)
        x15 = x13 + x14
        x16 = self.conv21(x15)
        x17 = self.conv22(x15)
        x18 = x16 + x17
        x19 = self.conv23(x18)
        x20 = self.conv24(x18)
        x21 = x19 + x20
        v38 = self.conv2(x1)
        v39 = self.conv1(x2)
        v40 = v38 + v39
        v41 = self.conv3(v40)
        v42 = self.conv4(v40)
        v43 = v41 + v42
        v44 = self.conv5(v43)
        v45 = self.conv6(v43)
        v46 = v44 + v45
        v47 = self.conv7(v46)
        v48 = self.conv8(v46)
        v49 = v47 + v48
        v50 = self.conv9(v49)
        v51 = self.conv10(v49)
        v52 = v50 + v51
        x1 = self.conv11(v52)
        x2 = self.conv12(v52)
        x3 = x1 + x2
        x4 = self.conv13(x3)
        x5 = self.conv14(x3)
        x6 = x4 + x5
        x7 = self.conv15(x6)
        x8 = self.conv16(x6)
        x9 = x7 + x8
        x10 = self.conv17(x9)
        x11 = self.conv18(x9)
        x12 = x10 + x11
        x13 = self.conv19(x12)
        x14 = self.conv20(x12)
        x15 = x13 + x14
        x16 = self.conv21(x15)
        x17 = self.conv22(x15)
        x18 = x16 + x17
        x19 = self.conv23(x18)
        x20 = self.conv24(x18)
        x21 = x19 + x20
        v53 = self.conv3(x1)
        v54 = self.conv4(x2)
        v55 = v53 + v54
        v56 = self.conv5(v55)
        v57 = self.conv6(v55)
        v58 = v56 + v57
        v59 = self.conv7(v58)
        v60 = self.conv8(v58)
        v61 = v59 + v60
        v62 = self.conv9(v61)
        v63 = self.conv10(v61)
        v64 = v62 + v63
        x1 = self.conv11(v64)
        x2 = self.conv12(v64)
        x3 = x1 + x2
        x4 = self.conv13(x3)
        x5 = self.conv14(x3)
        x6 = x4 + x5
        x7 = self.conv15(x6)
        x8 = self.conv16(x6)
        x9 = x7 + x8
        x10 = self.conv17(x9)
        x11 = self.conv18(x9)
        x12 = x10 + x11
        x13 = self.conv19(x12)
        x14 = self.conv20(x12)
        x15 = x13 + x14
        x16 = self.conv21(x15)
        x17 = self.conv22(x15)
        x18 = x16 + x17
        x19 = self.conv23(x18)
        x20 = self.conv24(x18)
        x21 = x19 + x20
        v65 = self.conv4(x1)
        v66 = self.conv3(x2)
        v67 = v65 + v66
        v68 = self.conv5(v67)
        v69 = self.conv6(v67)
        v70 = v68 + v69
        v71 = self.conv7(v70)
        v72 = self.conv8(v70)
        v73 = v71 + v72
        v74 = self.conv9(v73)
        v75 = self.conv10(v73)
        v76 = v74 + v75
        x1 = self.conv11(v76)
        x2 = self.conv12(v76)
        x3 = x1 + x2
        x4 = self.conv13(x3)
        x5 = self.conv14(x3)
        x6 = x4 + x5
        x7 = self.conv15(x6)
        x8 = self.conv16(x6)
        x9 = x7 + x8
        x10 = self.conv17(x9)
        x11 = self.conv18(x9)
        x12 = x10 + x11
        x13 = self.conv19(x12)
        x14 = self.conv20(x12)
        x15 = x13 + x14
        x16 = self.conv21(x15)
        x17 = self.conv22(x15)
        x18 = x16 + x17
        x19 = self.conv23(x18)
        x20 = self.conv24(x18)
        x21 = x19 + x20
        v77 = self.conv5(x1)
        v78 = self.conv6(x2)
        v79 = v77 + v78
        v80 = self.conv7(v79)
        v81 = self.conv8(v79)
        v82 = v80 + v81
        v83 = self.conv9(v82)
        v84 = self.conv10(v82)
        v85 = v83 + v84
        x1 = self.conv11(v85)
        x2 = self.conv12(v85)
        x3 = x1 + x2
        x4 = self.conv13(x3)
        x5 = self.conv14(x3)
        x6 = x4 + x5
        x7 = self.conv15(x6)
        x8 = self.conv16(x6)
        x9 = x7 + x8
        x10 = self.conv17(x9)
        x11 = self.conv18(x9)
        x12 = x10 + x11
        x13 = self.conv19(x12)
        x14 = self.conv20(x12)
        x15 = x13 + x14
        x16 = self.conv21(x15)
        x17 = self.conv22(x15)
        x18 = x16 + x17
        x19 = self.conv23(x18)
        x20 = self.conv24(x18)
        x21 = x19 + x20
        v86 = self.conv6(x1)
        v87 = self.conv5(x2)
        v88 = v86 + v87
        v89 = self.conv7(v88)
        v90 = self.conv8(v88)
        v91 = v89 + v90
        v92 = self.conv9(v91)
        v93 = self.conv10(v91)
        v94 = v92 + v93
        x1 = self.conv11(v94)
        x2 = self.conv12(v94)
        x3 = x1 + x2
        x4 = self.conv13(x3)
        x5 = self.conv14(x3)
        x6 = x4 + x5
        x7 = self.conv15(x6)
        x8 = self.conv16(x6)
        x9 = x7 + x8
        x10 = self.conv17(x9)
        x11 = self.conv18(x9)
        x12 = x10 + x11
        x13 = self.conv19(x12)
        x14 = self.conv20(x12)
        x15 = x13 + x14
        x16 = self.conv21(x15)
        x17 = self.conv22(x15)
        x18 = x16 + x17
        x19 = self.conv23(x18)
        x20 = self.conv24(x18)
        x21 = x19 + x20
        v95 = self.conv7(x1)
        v96 = self.conv8(x2)
        v97 = v95 + v96
        v98 = self.conv9(v97)
        v99 = self.conv10(v97)
        v100 = v98 + v99
        x1 = self.conv11(v100)
        x2 = self.conv12(v100)
        x3 = x1 + x2
        x4 = self.conv13(x3)
        x5 = self.conv14(x3)
        x6 = x4 + x5
        x7 = self.conv15(x6)
        x8 = self.conv16(x6)
        x9 = x7 + x8
        x10 = self.conv17(x9)
        x11 = self.conv18(x9)
        x12 = x10 + x11
        x13 = self.conv19(x12)
        x14 = self.conv20(x12)
        x15 = x13 + x14
        x16 = self.conv21(x15)
        x17 = self.conv22(x15)
        x18 = x16 + x17
        x19 = self.conv23(x18)
        x20 = self.conv24(x18)
        x21 = x19 + x20
        v101 = self.conv8(x1)
        v102 = self.conv9(x2)
        v103 = v101 + v102
        v104 = self.conv10(v103)
        x1 = self.conv11(v104)
        x2 = self.conv12(v104)
        x3 = x1 + x2
        x4 = self.conv13(x3)
        x5 = self.conv14(x3)
        x6 = x4 + x5
        x7 = self.conv15(x6)
        x8 = self.conv16(x6)
        x9 = x7 + x8
        x10 = self.conv17(x9)
        x11 = self.conv18(x9)
        x12 = x10 + x11
        x13 = self.conv19(x12)
        x14 = self.conv20(x12)
        x15 = x13 + x14
        x16 = self.conv21(x15)
        x17 = self.conv22(x15)
        x18 = x16 + x17
        x19 = self.conv23(x18)
        x20 = self.conv24(x18)
        x21 = x19 + x20
        v105 = self.conv9(x1)
        v106 = self.conv10(x2)
        v107 = v105 + v106
        x1 = self.conv11(v107)
        x2 = self.conv12(v107)
        x3 = x1 + x2
        x4 = self.conv13(x3)
        x5 = self.conv14(x3)
        x6 = x4 + x5
        x7 = self.conv15(x6)
        x8 = self.conv16(x6)
        