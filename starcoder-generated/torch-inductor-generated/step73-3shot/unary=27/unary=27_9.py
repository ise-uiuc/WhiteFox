
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv = torch.nn.Conv2d(832, 832, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(832, 832, 1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(832, 832, 1, stride=1, padding=0)
        self.conv4 = torch.nn.Conv2d(832, 832, 1, stride=1, padding=0)
        self.conv5 = torch.nn.Conv2d(832, 832, 1, stride=1, padding=0)
        self.conv6 = torch.nn.Conv2d(832, 832, 1, stride=1, padding=0)
        self.conv7 = torch.nn.Conv2d(832, 832, 1, stride=1, padding=0)
        self.conv8 = torch.nn.Conv2d(832, 832, 1, stride=1, padding=0)
        self.conv9 = torch.nn.Conv2d(832, 832, 1, stride=1, padding=0)
        self.conv10 = torch.nn.Conv2d(832, 832, 1, stride=1, padding=0)
        self.conv11 = torch.nn.Conv2d(832, 832, 1, stride=1, padding=0)
        self.conv12 = torch.nn.Conv2d(832, 832, 1, stride=1, padding=0)
        self.conv13 = torch.nn.Conv2d(832, 832, 1, stride=1, padding=0)
        self.conv14 = torch.nn.Conv2d(832, 832, 1, stride=1, padding=0)
        self.conv15 = torch.nn.Conv2d(832, 832, 1, stride=1, padding=0)
        self.conv16 = torch.nn.Conv2d(832, 832, 1, stride=1, padding=0)
        self.conv17 = torch.nn.Conv2d(832, 832, 1, stride=1, padding=0)
        self.conv18 = torch.nn.Conv2d(832, 832, 1, stride=1, padding=0)
        self.conv19 = torch.nn.Conv2d(832, 832, 1, stride=1, padding=0)
        self.conv20 = torch.nn.Conv2d(832, 832, 1, stride=1, padding=0)
        self.conv21 = torch.nn.Conv2d(832, 832, 1, stride=1, padding=0)
        self.conv22 = torch.nn.Conv2d(832, 832, 1, stride=1, padding=0)
        self.conv23 = torch.nn.Conv2d(832, 832, 1, stride=1, padding=0)
        self.conv24 = torch.nn.Conv2d(832, 832, 1, stride=1, padding=0)
        self.conv25 = torch.nn.Conv2d(832, 832, 1, stride=1, padding=0)
        self.conv26 = torch.nn.Conv2d(832, 832, 1, stride=1, padding=0)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, self.min)
        v3 = torch.clamp_max(v2, self.max)
        v4 = self.conv2(v3)
        v5 = torch.clamp_min(v4, self.min)
        v6 = torch.clamp_max(v5, self.max)
        v7 = self.conv3(v6)
        v8 = torch.clamp_min(v7, self.min)
        v9 = torch.clamp_max(v8, self.max)
        v10 = self.conv4(v9)
        v11 = torch.clamp_min(v10, self.min)
        v12 = torch.clamp_max(v11, self.max)
        v13 = self.conv5(v12)
        v14 = torch.clamp_min(v13, self.min)
        v15 = torch.clamp_max(v14, self.max)
        v16 = self.conv6(v15)
        v17 = torch.clamp_min(v16, self.min)
        v18 = torch.clamp_max(v17, self.max)
        v19 = self.conv7(v18)
        v20 = torch.clamp_min(v19, self.min)
        v21 = torch.clamp_max(v20, self.max)
        v22 = self.conv8(v21)
        v23 = torch.clamp_min(v22, self.min)
        v24 = torch.clamp_max(v23, self.max)
        v25 = self.conv9(v24)
        v26 = torch.clamp_min(v25, self.min)
        v27 = torch.clamp_max(v26, self.max)
        v28 = self.conv10(v27)
        v29 = torch.clamp_min(v28, self.min)
        v30 = torch.clamp_max(v29, self.max)
        v31 = self.conv11(v30)
        v32 = torch.clamp_min(v31, self.min)
        v33 = torch.clamp_max(v32, self.max)
        v34 = self.conv12(v33)
        v35 = torch.clamp_min(v34, self.min)
        v36 = torch.clamp_max(v35, self.max)
        v37 = self.conv13(v36)
        v38 = torch.clamp_min(v37, self.min)
        v39 = torch.clamp_max(v38, self.max)
        v40 = self.conv14(v39)
        v41 = torch.clamp_min(v40, self.min)
        v42 = torch.clamp_max(v41, self.max)
        v43 = self.conv15(v42)
        v44 = torch.clamp_min(v43, self.min)
        v45 = torch.clamp_max(v44, self.max)
        v46 = self.conv16(v45)
        v47 = torch.clamp_min(v46, self.min)
        v48 = torch.clamp_max(v47, self.max)
        v49 = self.conv17(v48)
        v50 = torch.clamp_min(v49, self.min)
        v51 = torch.clamp_max(v50, self.max)
        v52 = self.conv18(v51)
        v53 = torch.clamp_min(v52, self.min)
        v54 = torch.clamp_max(v53, self.max)
        v55 = self.conv19(v54)
        v56 = torch.clamp_min(v55, self.min)
        v57 = torch.clamp_max(v56, self.max)
        v58 = self.conv20(v57)
        v59 = torch.clamp_min(v58, self.min)
        v60 = torch.clamp_max(v59, self.max)
        v61 = self.conv21(v60)
        v62 = torch.clamp_min(v61, self.min)
        v63 = torch.clamp_max(v62, self.max)
        v64 = self.conv22(v63)
        v65 = torch.clamp_min(v64, self.min)
        v66 = torch.clamp_max(v65, self.max)
        v67 = self.conv23(v66)
        v68 = torch.clamp_min(v67, self.min)
        v69 = torch.clamp_max(v68, self.max)
        v70 = self.conv24(v69)
        v71 = torch.clamp_min(v70, self.min)
        v72 = torch.clamp_max(v71, self.max)
        v73 = self.conv25(v72)
        v74 = torch.clamp_min(v73, self.min)
        v75 = torch.clamp_max(v74, self.max)
        v76 = self.conv26(v75)
        v77 = torch.clamp_min(v76, self.min)
        v78 = torch.clamp_max(v77, self.max)
        return v78
min = 0.15
max = 0.5
# Inputs to the model
x1 = torch.randn(1, 832, 17, 17)
