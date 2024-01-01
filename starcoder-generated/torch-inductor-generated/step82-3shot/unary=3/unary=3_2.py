
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 1, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(1, 13, 1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(13, 9, 3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(9, 31, 3, stride=1, padding=1)
        self.conv5 = torch.nn.Conv2d(31, 8, 2, stride=2, padding=0)
        self.conv6 = torch.nn.Conv2d(8, 17, 3, stride=1, padding=1)
        self.conv7 = torch.nn.Conv2d(17, 11, 3, stride=1, padding=1)
        self.conv8 = torch.nn.Conv2d(11, 36, 3, stride=1, padding=1)
        self.conv9 = torch.nn.Conv2d(36, 4, 1, stride=1, padding=0)
        self.conv10 = torch.nn.Conv2d(4, 22, 3, stride=1, padding=1)
        self.conv11 = torch.nn.Conv2d(22, 7, 3, stride=1, padding=1)
        self.conv12 = torch.nn.Conv2d(7, 20, 2, stride=2, padding=0)
        self.conv13 = torch.nn.Conv2d(20, 16, 3, stride=1, padding=1)
        self.conv14 = torch.nn.Conv2d(16, 15, 3, stride=1, padding=1)
        self.conv15 = torch.nn.Conv2d(15, 16, 2, stride=2, padding=0)
        self.conv16 = torch.nn.Conv2d(16, 19, 3, stride=1, padding=1)
        self.conv17 = torch.nn.Conv2d(19, 15, 3, stride=1, padding=1)
        self.conv18 = torch.nn.Conv2d(15, 34, 2, stride=2, padding=0)
        self.conv19 = torch.nn.Conv2d(34, 10, 3, stride=1, padding=1)
        self.conv20 = torch.nn.Conv2d(10, 25, 1, stride=1, padding=0)
        self.conv21 = torch.nn.Conv2d(25, 31, 2, stride=2, padding=0)
        self.conv22 = torch.nn.Conv2d(31, 15, 5, stride=1, padding=2)
        self.conv23 = torch.nn.Conv2d(15, 7, 5, stride=1, padding=2)
        self.conv24 = torch.nn.Conv2d(7, 9, 3, stride=1, padding=1)
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
        v14 = v13 * 0.5
        v15 = v13 * 0.7071067811865476
        v16 = torch.erf(v15)
        v17 = v16 + 1
        v18 = v14 * v17
        v19 = self.conv4(v18)
        v20 = v19 * 0.5
        v21 = v19 * 0.7071067811865476
        v22 = torch.erf(v21)
        v23 = v22 + 1
        v24 = v20 * v23
        v25 = self.conv5(v24)
        v26 = v25 * 0.5
        v27 = v25 * 0.7071067811865476
        v28 = torch.erf(v27)
        v29 = v28 + 1
        v30 = v26 * v29
        v31 = self.conv6(v30)
        v32 = v31 * 0.5
        v33 = v31 * 0.7071067811865476
        v34 = torch.erf(v33)
        v35 = v34 + 1
        v36 = v32 * v35
        v37 = self.conv7(v36)
        v38 = v37 * 0.5
        v39 = v37 * 0.7071067811865476
        v40 = torch.erf(v39)
        v41 = v40 + 1
        v42 = v38 * v41
        v43 = self.conv8(v42)
        v44 = v43 * 0.5
        v45 = v43 * 0.7071067811865476
        v46 = torch.erf(v45)
        v47 = v46 + 1
        v48 = v44 * v47
        v49 = self.conv9(v48)
        v50 = v49 * 0.5
        v51 = v49 * 0.7071067811865476
        v52 = torch.erf(v51)
        v53 = v52 + 1
        v54 = v50 * v53
        v55 = self.conv10(v54)
        v56 = v55 * 0.5
        v57 = v55 * 0.7071067811865476
        v58 = torch.erf(v57)
        v59 = v58 + 1
        v60 = v56 * v59
        v61 = self.conv11(v60)
        v62 = v61 * 0.5
        v63 = v61 * 0.7071067811865476
        v64 = torch.erf(v63)
        v65 = v64 + 1
        v66 = v62 * v65
        v67 = self.conv12(v66)
        v68 = v67 * 0.5
        v69 = v67 * 0.7071067811865476
        v70 = torch.erf(v69)
        v71 = v70 + 1
        v72 = v68 * v71
        v73 = self.conv13(v72)
        v74 = v73 * 0.5
        v75 = v73 * 0.7071067811865476
        v76 = torch.erf(v75)
        v77 = v76 + 1
        v78 = v74 * v77
        v79 = self.conv14(v78)
        v80 = v79 * 0.5
        v81 = v79 * 0.7071067811865476
        v82 = torch.erf(v81)
        v83 = v82 + 1
        v84 = v80 * v83
        v85 = self.conv15(v84)
        v86 = v85 * 0.5
        v87 = v85 * 0.7071067811865476
        v88 = torch.erf(v87)
        v89 = v88 + 1
        v90 = v86 * v89
        v91 = self.conv16(v90)
        v92 = v91 * 0.5
        v93 = v91 * 0.7071067811865476
        v94 = torch.erf(v93)
        v95 = v94 + 1
        v96 = v92 * v95
        v97 = self.conv17(v96)
        v98 = v97 * 0.5
        v99 = v97 * 0.7071067811865476
        v100 = torch.erf(v99)
        v101 = v100 + 1
        v102 = v98 * v101
        v103 = self.conv18(v102)
        v104 = v103 * 0.5
        v105 = v103 * 0.7071067811865476
        v106 = torch.erf(v105)
        v107 = v106 + 1
        v108 = v104 * v107
        v109 = self.conv19(v108)
        v110 = v109 * 0.5
        v111 = v109 * 0.7071067811865476
        v112 = torch.erf(v111)
        v113 = v112 + 1
        v114 = v110 * v113
        v115 = self.conv20(v114)
        v116 = v115 * 0.5
        v117 = v115 * 0.7071067811865476
        v118 = torch.erf(v117)
        v119 = v118 + 1
        v120 = v116 * v119
        v121 = self.conv21(v120)
        v122 = v121 * 0.5
        v123 = v121 * 0.7071067811865476
        v124 = torch.erf(v123)
        v125 = v124 + 1
        v126 = v122 * v125
        v127 = self.conv22(v126)
        v128 = v127 * 0.5
        v129 = v127 * 0.7071067811865476
        v130 = torch.erf(v129)
        v131 = v130 + 1
        v132 = v128 * v131
        v133 = self.conv23(v132)
        v134 = v133 * 0.5
        v134 = v133 * 0.7071067811865476
        v135 = torch.erf(v134)
        v136 = v135 + 1
        v137 = v133 * v136
        v138 = self.conv24(v137)
        return v138
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
