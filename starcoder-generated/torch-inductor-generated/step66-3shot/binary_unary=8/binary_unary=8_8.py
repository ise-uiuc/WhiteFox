
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(1, 3, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.conv1(x1)
        v4 = self.conv2(v3)
        v5 = self.conv2(v1)
        v6 = self.conv1(x1)
        v7 = self.conv2(v6)
        v8 = self.conv1(x1)
        v9 = self.conv2(v8)
        v10 = self.conv2(v3)
        v11 = self.conv1(x1)
        v12 = self.conv2(v11)
        v13 = self.conv2(v8)
        v14 = self.conv1(x1)
        v15 = self.conv2(v14)
        v16 = self.conv1(x1)
        v17 = self.conv2(v16)
        v18 = self.conv2(v11)
        v19 = self.conv2(v6)
        v20 = self.conv2(v1)
        v21 = self.conv1(x1)
        v22 = self.conv2(v2)
        v23 = v21 + v22
        v24 = self.conv2(v5)
        v25 = self.conv1(x1)
        v26 = self.conv2(v7)
        v27 = v25 + v26
        v28 = self.conv2(v20)
        v29 = self.conv2(v3)
        v30 = v27 + v28 + v29
        v31 = self.conv2(v15)
        v32 = self.conv1(x1)
        v33 = self.conv2(v4)
        v34 = self.conv1(x1)
        v35 = self.conv2(v9)
        v36 = v34 + v35
        v37 = self.conv2(v9)
        v38 = self.conv1(x1)
        v39 = self.conv2(v12)
        v40 = v38 + v39
        v41 = self.conv2(v12)
        v42 = self.conv2(v3)
        v43 = self.conv2(v8)
        v44 = self.conv2(v16)
        v45 = v42 + v43 + v44
        v46 = self.conv2(v19)
        v47 = self.conv1(x1)
        v48 = self.conv2(v23)
        v49 = self.conv1(x1)
        v50 = self.conv2(v24)
        v51 = v49 + v50
        v52 = self.conv2(v27)
        v53 = self.conv1(x1)
        v54 = self.conv2(v28)
        v55 = v53 + v54
        v56 = self.conv2(v30)
        v57 = self.conv2(v33)
        v58 = self.conv2(v36)
        v59 = self.conv2(v39)
        v60 = self.conv2(v42)
        v61 = self.conv2(x1)
        v62 = self.conv2(v61)
        v63 = v51 + v52 + v53 + v54 + v55 + v58 + v59
        v64 = self.conv2(v60)
        v65 = self.conv1(x1)
        v66 = self.conv2(v65)
        v67 = self.conv1(x1)
        v68 = self.conv2(v48)
        v69 = self.conv2(v40)
        v70 = self.conv2(v30)
        v71 = self.conv1(x1)
        v72 = self.conv2(v32)
        v73 = self.conv2(v37)
        v74 = self.conv2(v40)
        v75 = self.conv2(v43)
        v76 = self.conv2(v1)
        v77 = self.conv2(v44)
        v78 = self.conv2(v12)
        v79 = self.conv2(v36)
        v80 = self.conv1(x1)
        v81 = self.conv2(v47)
        v82 = self.conv1(x1)
        v83 = self.conv2(v45)
        v84 = self.conv2(v42)
        v85 = self.conv2(v30)
        v86 = self.conv2(x1)
        v87 = self.conv2(v63)
        v88 = self.conv2(v66)
        v89 = self.conv2(v62)
        v90 = self.conv2(v56)
        v91 = v67 + v68 + v69 + v71 + v73 + v76 + v77 + v78 + v79 + v80
        v92 = self.conv2(v60)
        v93 = self.conv2(v72)
        v94 = v82 + v83 + v84 + v86 + v88 + v89
        v95 = self.conv2(v90)
        v96 = v86 + v80 + v81
        v97 = self.conv2(v85)
        v98 = v91 + v92 + v93 + v94 + v96
        return v95 + v97 + v98
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
