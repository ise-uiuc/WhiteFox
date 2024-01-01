
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=0)
        self.conv4 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=0)
        self.conv5 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=0)
    def forward(self, x):
        v0 = x
        v1 = self.conv1(v0)
        v2 = v0 > 0
        v3 = v0 * 0.3
        v4 = torch.where(v2, v0, v3)
        v5 = v4 + v1
        v6 = v5 > 0
        v7 = v5 * 0.1
        v8 = torch.where(v6, v5, v7)
        v9 = v8 + v5
        v10 = v5 > 0
        v11 = v5 * -1.0
        v12 = torch.where(v10, v5, v11)
        v13 = v12 + v8
        v14 = v13 > 0
        v15 = v13 * 0.5
        v16 = torch.where(v14, v13, v15)
        v17 = self.conv2(v16)
        v18 = v16 > 0
        v19 = v16 * 0.3
        v20 = torch.where(v18, v16, v19)
        v21 = v20 + v17
        v22 = v21 > 0
        v23 = v21 * 0.1
        v24 = torch.where(v22, v21, v23)
        v25 = v24 + v21
        v26 = v21 > 0
        v27 = v21 * -1.0
        v28 = torch.where(v26, v21, v27)
        v29 = v28 + v24
        v30 = v29 > 0
        v31 = v29 * 0.5
        v32 = torch.where(v30, v29, v31)
        v33 = self.conv3(v32)
        v34 = v32 > 0
        v35 = v32 * 0.3
        v36 = torch.where(v34, v32, v35)
        v37 = v36 + v33
        v38 = v37 > 0
        v39 = v37 * 0.1
        v40 = torch.where(v38, v37, v39)
        v41 = v40 + v37
        v42 = v37 > 0
        v43 = v37 * -1.0
        v44 = torch.where(v42, v37, v43)
        v45 = v44 + v40
        v46 = v45 > 0
        v47 = v45 * 0.5
        v48 = torch.where(v46, v45, v47)
        v49 = self.conv4(v48)
        v50 = v48 > 0
        v51 = v48 * 0.3
        v52 = torch.where(v50, v48, v51)
        v53 = v52 + v49
        v54 = v53 > 0
        v55 = v53 * 0.1
        v56 = torch.where(v54, v53, v55)
        v57 = v56 + v53
        v58 = v53 > 0
        v59 = v53 * -1.0
        v60 = torch.where(v58, v53, v59)
        v61 = v60 + v56)
        v62 = v61 > 0
        v63 = v61 * 0.5
        v64 = torch.where(v62, v61, v63)
        v65 = self.conv5(v64)
        v66 = v64 > 0
        v67 = v64 * 0.3
        v68 = torch.where(v66, v64, v67)
        v69 = v68 + v65
        v70 = v69 > 0
        v71 = v69 * 0.1
        v72 = torch.where(v70, v69, v71)
        v73 = v72 + v69
        v74 = v69 > 0
        v75 = v69 * -1.0
        v76 = torch.where(v74, v69, v75)
        v77 = v76 + v72
        v78 = v77 > 0
        v79 = v77 * 0.5
        v80 = torch.where(v78, v77, v79)
        v81 = v80 + x
        return v81
# Inputs to the model
x1 = torch.randn(1, 8, 64, 64)
