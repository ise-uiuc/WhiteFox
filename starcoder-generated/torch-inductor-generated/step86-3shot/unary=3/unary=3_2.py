
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tensordot = torch.nn.functional.tensordot
    def forward(self, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13):
        v1 = self.tensordot(x1, x2, dims=0)
        v2 = self.tensordot(v1 * 0.5, x1, dims=0)
        v3 = self.tensordot(v1 * 0.7071067811865476, x1, dims=0)
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = self.tensordot(v5, x1, dims=0)
        v7 = v2 * v6
        v8 = self.tensordot(v3, x2, dims=0)
        v9 = self.tensordot(v8 * 0.5, x2, dims=0)
        v10 = self.tensordot(v8 * 0.7071067811865476, x2, dims=0)
        v11 = torch.erf(v10)
        v12 = v11 + 1
        v13 = self.tensordot(v12, x2, dims=0)
        v14 = v9 * v13
        v15 = self.tensordot(v10, x3, dims=0)
        v16 = self.tensordot(v15 * 0.5, x3, dims=0)
        v17 = self.tensordot(v15 * 0.7071067811865476, x3, dims=0)
        v18 = torch.erf(v17)
        v19 = v18 + 1
        v20 = self.tensordot(v19, x3, dims=0)
        v21 = v16 * v20
        v22 = self.tensordot(v18, x4, dims=0)
        v23 = self.tensordot(v22 * 0.5, x4, dims=0)
        v24 = self.tensordot(v22 * 0.7071067811865476, x4, dims=0)
        v25 = torch.erf(v24)
        v26 = v25 + 1
        v27 = self.tensordot(v26, x4, dims=0)
        v28 = v23 * v27
        v29 = self.tensordot(v25, x5, dims=0)
        v30 = self.tensordot(v29 * 0.5, x5, dims=0)
        v31 = self.tensordot(v29 * 0.7071067811865476, x5, dims=0)
        v32 = torch.erf(v31)
        v33 = v32 + 1
        v34 = self.tensordot(v33, x5, dims=0)
        v35 = v30 * v34
        v36 = self.tensordot(v32, x6, dims=0)
        v37 = self.tensordot(v36 * 0.5, x6, dims=0)
        v38 = self.tensordot(v36 * 0.7071067811865476, x6, dims=0)
        v39 = torch.erf(v38)
        v40 = v39 + 1
        v41 = self.tensordot(v40, x6, dims=0)
        v42 = v37 * v41
        v43 = self.tensordot(v39, x7, dims=0)
        v44 = self.tensordot(v43 * 0.5, x7, dims=0)
        v45 = self.tensordot(v43 * 0.7071067811865476, x7, dims=0)
        v46 = torch.erf(v45)
        v47 = v46 + 1
        v48 = self.tensordot(v47, x7, dims=0)
        v49 = v44 * v48
        v50 = self.tensordot(v46, x8, dims=0)
        v51 = self.tensordot(v50 * 0.5, x8, dims=0)
        v52 = self.tensordot(v50 * 0.7071067811865476, x8, dims=0)
        v53 = torch.erf(v52)
        v54 = v53 + 1
        v55 = self.tensordot(v54, x8, dims=0)
        v56 = v51 * v55
        v57 = self.tensordot(v53, x9, dims=0)
        v58 = self.tensordot(v57 * 0.5, x9, dims=0)
        v59 = self.tensordot(v57 * 0.7071067811865476, x9, dims=0)
        v60 = torch.erf(v59)
        v61 = v60 + 1
        v62 = self.tensordot(v61, x9, dims=0)
        v63 = v58 * v62
        v64 = self.tensordot(v60, x10, dims=0)
        v65 = self.tensordot(v64 * 0.5, x10, dims=0)
        v66 = self.tensordot(v64 * 0.7071067811865476, x10, dims=0)
        v67 = torch.erf(v66)
        v68 = v67 + 1
        v69 = self.tensordot(v68, x10, dims=0)
        v70 = v65 * v69
        v71 = self.tensordot(v67, x11, dims=0)
        v72 = self.tensordot(v71 * 0.5, x11, dims=0)
        v73 = self.tensordot(v71 * 0.7071067811865476, x11, dims=0)
        v74 = torch.erf(v73)
        v75 = v74 + 1
        v76 = self.tensordot(v75, x11, dims=0)
        v77 = v72 * v76
        v78 = self.tensordot(v74, x12, dims=0)
        v79 = self.tensordot(v78 * 0.5, x12, dims=0)
        v80 = self.tensordot(v78 * 0.7071067811865476, x12, dims=0)
        v81 = torch.erf(v80)
        v82 = v81 + 1
        v83 = self.tensordot(v82, x12, dims=0)
        v84 = v79 * v83
        v85 = self.tensordot(v81, x13, dims=0)
        v86 = self.tensordot(v85 * 0.5, x13, dims=0)
        v87 = self.tensordot(v85 * 0.7071067811865476, x13, dims=0)
        v88 = torch.erf(v87)
        v89 = v88 + 1
        v90 = self.tensordot(v89, x13, dims=0)
        v91 = v86 * v90
        v92 = v77 * 0.5
        v93 = v77 * 0.7071067811865476
        v94 = torch.erf(v93)
        v95 = v94 + 1
        v96 = v92 * v95
        v97 = v94 * 0.5
        v98 = v94 + 1
        v99 = v97 * v98
        v100 = v96 * v99
        return v100 + v100
# Inputs to the model
x1 = torch.randn(199, 54, 13, 23)
x2 = torch.randn(1, 13, 29, 17)
x3 = torch.randn(7, 29, 41, 9)
x4 = torch.randn(9, 20, 48, 15)
x5 = torch.randn(15, 17, 38, 5)
x6 = torch.randn(20, 28, 40, 23)
x7 = torch.randn(17, 4, 22, 7)
x8 = torch.randn(5, 48, 26, 1)
x9 = torch.randn(6, 12, 30, 27)
x10 = torch.randn(10, 15, 7, 15)
x11 = torch.randn(17, 48, 39, 27)
x12 = torch.randn(4, 27, 8, 11)
x13 = torch.randn(18, 70, 35, 49)
