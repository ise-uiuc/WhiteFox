
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 1, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(3, 64, 1, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(1, 64, 1, stride=1, padding=3)
        self.conv4 = torch.nn.Conv2d(1, 64, 1, stride=1, padding=1)
        self.conv5 = torch.nn.Conv2d(1, 128, 1, stride=1, padding=3)
        self.conv6 = torch.nn.Conv2d(1, 128, 1, stride=1, padding=1)
        self.conv7 = torch.nn.Conv2d(1, 256, 1, stride=1, padding=3)
        self.conv8 = torch.nn.Conv2d(1, 256, 1, stride=1, padding=1)
        self.conv9 = torch.nn.Conv2d(1, 512, 1, stride=1, padding=3)
        self.conv10 = torch.nn.Conv2d(1, 512, 1, stride=1, padding=1)
        self.conv11 = torch.nn.Conv2d(1, 1024, 1, stride=1, padding=3)
        self.conv12 = torch.nn.Conv2d(1, 1024, 1, stride=1, padding=1)
        self.conv13 = torch.nn.Conv2d(1, 128, 1, stride=1, padding=144)
        self.conv14 = torch.nn.Conv2d(1, 128, 1, stride=1, padding=192)
        self.conv15 = torch.nn.Conv2d(1, 128, 1, stride=1, padding=61)
        self.conv16 = torch.nn.Conv2d(1, 128, 1, stride=1, padding=103)
        self.conv17 = torch.nn.Conv2d(1, 128, 1, stride=1, padding=68)
        self.conv18 = torch.nn.Conv2d(1, 128, 1, stride=1, padding=54)
        self.conv19 = torch.nn.Conv2d(1, 128, 1, stride=1, padding=188)
        self.conv20 = torch.nn.Conv2d(1, 64, 1, stride=1, padding=126)
        self.conv21 = torch.nn.Conv2d(3, 64, 1, stride=1, padding=32)
        self.conv22 = torch.nn.Conv2d(1, 64, 1, stride=1, padding=10)
        self.conv23 = torch.nn.Conv2d(1, 64, 1, stride=1, padding=13)
        self.conv24 = torch.nn.Conv2d(1, 64, 1, stride=1, padding=118)
        self.conv25 = torch.nn.Conv2d(1, 64, 1, stride=1, padding=133)
        self.conv26 = torch.nn.Conv2d(1, 64, 1, stride=1, padding=153)
        self.conv27 = torch.nn.Conv2d(1, 64, 1, stride=1, padding=167)
        self.conv28 = torch.nn.Conv2d(1, 64, 1, stride=1, padding=35)
        self.conv29 = torch.nn.Conv2d(1, 64, 1, stride=1, padding=73)
        self.conv30 = torch.nn.Conv2d(1, 64, 1, stride=1, padding=155)
        self.conv31 = torch.nn.Conv2d(1, 64, 1, stride=1, padding=27)
        self.conv32 = torch.nn.Conv2d(1, 128, 1, stride=1, padding=6)
        self.conv33 = torch.nn.Conv2d(1, 128, 1, stride=1, padding=137)
        self.conv34 = torch.nn.Conv2d(1, 128, 1, stride=1, padding=41)
        self.conv35 = torch.nn.Conv2d(1, 128, 1, stride=1, padding=38)
        self.conv36 = torch.nn.Conv2d(1, 64, 1, stride=1, padding=31)
        self.conv37 = torch.nn.Conv2d(1, 64, 1, stride=1, padding=175)
        self.conv38 = torch.nn.Conv2d(1, 128, 1, stride=1, padding=92)
        self.conv39 = torch.nn.Conv2d(1, 128, 1, stride=1, padding=94)
        self.conv40 = torch.nn.Conv2d(1, 128, 1, stride=1, padding=141)
        self.conv41 = torch.nn.Conv2d(1, 64, 1, stride=1, padding=136)
        self.conv42 = torch.nn.Conv2d(1, 64, 1, stride=1, padding=68)
        self.conv43 = torch.nn.Conv2d(1, 256, 1, stride=1, padding=76)
        self.conv44 = torch.nn.Conv2d(1, 256, 1, stride=1, padding=105)
        self.conv45 = torch.nn.Conv2d(1, 64, 1, stride=1, padding=131)
        self.conv46 = torch.nn.Conv2d(1, 512, 1, stride=1, padding=152)
        self.conv47 = torch.nn.Conv2d(1, 64, 1, stride=1, padding=165)
        self.conv48 = torch.nn.Conv2d(1, 128, 1, stride=1, padding=54)
        self.conv49 = torch.nn.Conv2d(1, 1024, 1, stride=1, padding=63)
        self.conv50 = torch.nn.Conv2d(1, 64, 1, stride=1, padding=154)
        self.conv51 = torch.nn.Conv2d(1, 64, 1, stride=1, padding=107)
        self.conv52 = torch.nn.Conv2d(1, 256, 1, stride=1, padding=91)
        self.conv53 = torch.nn.Conv2d(1, 512, 1, stride=1, padding=128)
        self.conv54 = torch.nn.Conv2d(1, 64, 1, stride=1, padding=16)
        self.conv55 = torch.nn.Conv2d(1, 64, 1, stride=1, padding=131)
        self.conv56 = torch.nn.Conv2d(1, 64, 1, stride=1, padding=13)
        self.conv57 = torch.nn.Conv2d(1, 64, 1, stride=1, padding=51)
        self.conv58 = torch.nn.Conv2d(1, 128, 1, stride=1, padding=40)
        self.conv59 = torch.nn.Conv2d(1, 64, 1, stride=1, padding=113)
        self.conv60 = torch.nn.Conv2d(1, 512, 1, stride=1, padding=67)
        self.conv61 = torch.nn.Conv2d(1, 64, 1, stride=1, padding=159)
        self.conv62 = torch.nn.Conv2d(1, 512, 1, stride=1, padding=130)
        self.conv63 = torch.nn.Conv2d(1, 64, 1, stride=1, padding=61)
        self.conv64 = torch.nn.Conv2d(1, 256, 1, stride=1, padding=191)
        self.conv65 = torch.nn.Conv2d(1, 128, 1, stride=1, padding=173)
        self.conv66 = torch.nn.Conv2d(1, 128, 1, stride=1, padding=151)
        self.conv67 = torch.nn.Conv2d(3, 64, 1, stride=1, padding=5)
        self.conv68 = torch.nn.Conv2d(1, 64, 1, stride=1, padding=92)
        self.conv69 = torch.nn.Conv2d(1, 64, 1, stride=1, padding=21)
        self.conv70 = torch.nn.Conv2d(1, 64, 1, stride=1, padding=140)
        self.conv71 = torch.nn.Conv2d(1, 64, 1, stride=1, padding=11)
        self.conv72 = torch.nn.Conv2d(1, 128, 1, stride=1, padding=106)
        self.conv73 = torch.nn.Conv2d(1, 64, 1, stride=1, padding=88)
        self.conv74 = torch.nn.Conv2d(1, 64, 1, stride=1, padding=56)
        self.conv75 = torch.nn.Conv2d(1, 64, 1, stride=1, padding=135)
        self.conv76 = torch.nn.Conv2d(1, 64, 1, stride=1, padding=182)
        self.conv77 = torch.nn.Conv2d(1, 64, 1, stride=1, padding=189)
        self.conv78 = torch.nn.Conv2d(1, 64, 1, stride=1, padding=63)
        self.conv79 = torch.nn.Conv2d(1, 64, 1, stride=1, padding=87)
        self.conv80 = torch.nn.Conv2d(1, 64, 1, stride=1, padding=27)
        self.conv81 = torch.nn.Conv2d(1, 128, 1, stride=1, padding=162)
        self.conv82 = torch.nn.Conv2d(1, 64, 1, stride=1, padding=141)
    def forward(self, x1):
        v1 = self.conv1(x1)
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
        v121 = self.conv21(x1)
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
        v135 = v133 * 0.7071067811865476
        v136 = torch.erf(v135)
        v137 = v136 + 1
        v138 = v134 * v137
        v139 = self.conv24(v138)
        v140 = v139 * 0.5
        v141 = v139 * 0.7071067811865476
        v142 = torch.erf(v141)
        v143 = v142 + 1
      