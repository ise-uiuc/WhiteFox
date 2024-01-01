
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 1, 3, stride=1, padding=0)
    def forward(self, x7):
        v0 = x7.shape
        y0 = torch.randint(low=-8, high=8, size=(1,), device=torch.device('cpu'))
        z0 = torch.randint(low=1, high=4, size=(1,), device=torch.device('cpu'))
        v1 = torch.eq(y0, 0)
        v2 = torch.bitwise_and(v1, z0)
        v3 = torch.eq(v2, 2)
        z0_ = (z0, 2)
        v3_ = v3
        v4 = v3_ or v2
        v5 = torch.mul(y0, v4)
        v6 = torch.mul(z0_, v5)
        v7 = v6.shape
        t0 = torch.Size([1])
        t1 = v6.stride()
        v8 = torch.eq(t0, t1)
        y1 = torch.Size(t0)
        v9 = torch.Tensor(y1).uniform_()
        v10 = torch.greater(v9, 0.5)
        v11 = v10.view(v7)
        v12 = v6.permute(v11)
        v13 = v12.contiguous()
        v14 = v13.view(1, -1, 2, 1)
        v15 = v14.size()
        t2 = torch.Size([1])
        v16 = torch.arange(0, (z0_), 2, out=torch.LongTensor())
        v17 = v16.tolist()
        y2 = torch.Size([2])
        v18 = torch.randint(low=-8, high=8, size=(y2,), device=torch.device('cpu'))
        z1 = torch.randint(low=1, high=9, size=(y2,), device=torch.device('cpu'))
        v19 = torch.eq(z1, 0)
        v20 = torch.bitwise_and(v19, z0)
        v21 = torch.eq(v20, 2)
        z1_ = (z1, 2)
        v21_ = v21
        v22 = v21_ or v20
        v23 = torch.mul(z1_, v22)
        v24 = torch.mul(v18, v23)
        v25 = v24.shape
        t3 = torch.Size([2])
        t4 = v24.stride()
        v26 = torch.eq(t3, t4)
        y3 = torch.Size(t3)
        v27 = torch.Tensor(y3).uniform_()
        v28 = torch.greater(v27, 0.5)
        v29 = v28.view(v25)
        v30 = v24.permute(v29)
        v31 = v30.contiguous()
        v32 = v31.view(1, -1, 2, 2)
        v33 = v14 * v32
        v34 = v33.size()
        v35 = v34[0]
        v36 = v34[1]
        t5 = torch.Size([1])
        t6 = torch.Size([2])
        v37 = torch.arange(0, (z0_), 2, out=torch.LongTensor())
        v38 = v37.tolist()
        y4 = torch.Size([2])
        v39 = torch.randint(low=-8, high=8, size=(y4,), device=torch.device('cpu'))
        z2 = torch.randint(low=1, high=4, size=(y4,), device=torch.device('cpu'))
        v40 = torch.eq(z2, 0)
        v41 = torch.bitwise_and(v40, z0)
        v42 = torch.eq(v41, 2)
        z2_ = (z2, 2)
        v42_ = v42
        v43 = v42_ or v41
        v44 = torch.mul(z2_, v43)
        v45 = torch.mul(v39, v44)
        v46 = v45.shape
        t7 = torch.Size([2])
        t8 = v45.stride()
        v47 = torch.eq(t7, t8)
        y5 = torch.Size(t7)
        v48 = torch.Tensor(y5).uniform_()
        v49 = torch.greater(v48, 0.5)
        v50 = v49.view(v46)
        v51 = v45.permute(v50)
        v52 = v51.contiguous()
        v53 = v52.view(1, -1, 2, 1)
        v54 = v15[0]
        v55 = v15[1]
        t9 = torch.Size([1])
        t10 = torch.Size([2])
        v56 = torch.arange(0, (z0_), 2, out=torch.LongTensor())
        v57 = v56.tolist()
        y6 = torch.Size([2])
        v58 = torch.randint(low=-8, high=8, size=(y6,), device=torch.device('cpu'))
        z3 = torch.randint(low=1, high=9, size=(y6,), device=torch.device('cpu'))
        v59 = torch.eq(z3, 0)
        v60 = torch.bitwise_and(v59, z0)
        v61 = torch.eq(v60, 2)
        z3_ = (z3, 2)
        v61_ = v61
        v62 = v61_ or v60
        v63 = torch.mul(z3_, v62)
        v64 = torch.mul(v58, v63)
        v65 = v64.shape
        t11 = torch.Size([2])
        t12 = v64.stride()
        v66 = torch.eq(t11, t12)
        y7 = torch.Size(t11)
        v67 = torch.Tensor(y7).uniform_()
        v68 = torch.greater(v67, 0.5)
        v69 = v68.view(v65)
        v70 = v64.permute(v69)
        v71 = v70.contiguous()
        v72 = v71.view(1, -1, 2, 2)
        v73 = v54 + v72
        v74 = v73.size()
        v75 = v74[0]
        v76 = v74[1]
        t13 = torch.Size([1])
        t14 = torch.Size([2])
        v77 = torch.arange(0, (z0_), 2, out=torch.LongTensor())
        v78 = v77.tolist()
        y8 = torch.Size([2])
        v79 = torch.randint(low=-8, high=8, size=(y8,), device=torch.device('cpu'))
        z4 = torch.randint(low=1, high=4, size=(y8,), device=torch.device('cpu'))
        v80 = torch.eq(z4, 0)
        v81 = torch.bitwise_and(v80, z0)
        v82 = torch.eq(v81, 2)
        z4_ = (z4, 2)
        v82_ = v82
        v83 = v82_ or v81
        v84 = torch.mul(z4_, v83)
        v85 = torch.mul(v79, v84)
        v86 = v85.shape
        t15 = torch.Size([2])
        t16 = v85.stride()
        v87 = torch.eq(t15, t16)
        y9 = torch.Size(t15)
        v88 = torch.Tensor(y9).uniform_()
        v89 = torch.greater(v88, 0.5)
        v90 = v89.view(v86)
        v91 = v85.permute(v90)
        v92 = v91.contiguous()
        v93 = v92.view(1, -1, 2, 1)
        v94 = v35 + v75
        v95 = v94 * v76
        v96 = v95.size()
        v97 = v96[0]
        t17 = torch.Size([1])
        v97_ = v76
        v98 = float(v94)
        v99 = float(v97_)
        v100 = float(v87)
        v101 = float(v100)
        t18 = torch.Size([2])
        v102 = torch.randint(low=-8, high=8, size=(y8,), device=torch.device('cpu'))
        z5 = torch.randint(low=1, high=4, size=(y8,), device=torch.device('cpu'))
        v103 = torch.eq(z5, 0)
        v104 = torch.bitwise_and(v103, z0)
        v105 = torch.eq(v104, 2)
        z5_ = (z5, 2)
        v105_ = v105
        v106 = v105_ or v104
        v107 = torch.mul(z5_, v106)
        v108 = torch.mul(v102, v107)
        v109 = v108.shape
        t19 = torch.Size([2])
        t20 = v108.stride()
        v110 = torch.eq(t19, t20)
        y10 = torch.Size(t19)
        v111 = torch.Tensor(y10).uniform_()
        v112 = torch.greater(v111, 0.5)
        v113 = v112.view(v109)
        v114 = v108.permute(v113)
        v115 = v114.contiguous()
        v116 = v115.view(1, -1, 2, 1)
        v117 = v97 * v116
        v118 = v117.size()
        v119 = v118[0]
        v120 = v118[1]
        v121 = torch.mul(v89, v110)
        v122 = v121.view(v97_)
        v123 = v122.permute(v119)
        v124 = v123.contiguous()
        v125 = v124.view(1, v119, v101, 2, 1)
        v126 = v98 * v125
        v127 = v126.size()
        v128 = v127[0]
        v129 = v127[1]
        v130 = v128 * v129
        v131 = v130.size()
        v132 = v131[0]
        v133 = v131[1]
        v134 = v98 * v66
        v135 = v134.view(v101, v133, v132, v33)
        v136 = v99 * v135
        return v136
# Inputs to the model
x7 = torch.randn(1, 4, 32, 32)
