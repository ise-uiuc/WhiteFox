
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 1, stride=1, padding=26)
        self.conv2 = torch.nn.Conv2d(32, 32, 1, stride=1, padding=12)
        self.conv3 = torch.nn.Conv2d(32, 32, 1, stride=1, padding=7)
        self.conv4 = torch.nn.Conv2d(32, 32, 1, stride=1, padding=12)
        self.conv5 = torch.nn.Conv2d(32, 32, 1, stride=1, padding=7)
        self.conv6 = torch.nn.Conv2d(32, 32, 1, stride=1, padding=13)
        self.conv7 = torch.nn.Conv2d(32, 32, 1, stride=1, padding=7)
        self.conv8 = torch.nn.Conv2d(32, 32, 1, stride=1, padding=12)
        self.conv9 = torch.nn.Conv2d(32, 32, 1, stride=1, padding=15)
        self.conv10 = torch.nn.Conv2d(32, 32, 1, stride=1, padding=7)
        self.conv11 = torch.nn.Conv2d(32, 32, 1, stride=1, padding=6)
        self.conv12 = torch.nn.Conv2d(32, 32, 1, stride=1, padding=12)
        self.conv13 = torch.nn.Conv2d(32, 32, 1, stride=1, padding=10)
        self.conv14 = torch.nn.Conv2d(32, 32, 1, stride=1, padding=16)
        self.conv15 = torch.nn.Conv2d(32, 32, 1, stride=1, padding=9)
        self.conv16 = torch.nn.Conv2d(32, 32, 1, stride=1, padding=5)
        self.conv17 = torch.nn.Conv2d(32, 32, 1, stride=1, padding=6)
        self.conv18 = torch.nn.Conv2d(32, 32, 1, stride=1, padding=11)
        self.conv19 = torch.nn.Conv2d(32, 32, 1, stride=1, padding=9)
        self.conv20 = torch.nn.Conv2d(32, 32, 1, stride=1, padding=6)
        self.conv21 = torch.nn.Conv2d(32, 32, 1, stride=1, padding=11)
        self.conv22 = torch.nn.Conv2d(32, 32, 1, stride=1, padding=12)
        self.conv23 = torch.nn.Conv2d(32, 32, 1, stride=1, padding=13)
        self.conv24 = torch.nn.Conv2d(32, 32, 1, stride=1, padding=6)
        self.conv25 = torch.nn.Conv2d(32, 32, 1, stride=1, padding=12)
        self.conv26 = torch.nn.Conv2d(32, 32, 1, stride=1, padding=10)
        self.conv27 = torch.nn.Conv2d(32, 32, 1, stride=1, padding=6)
        self.conv28 = torch.nn.Conv2d(32, 32, 1, stride=1, padding=13)
        self.conv29 = torch.nn.Conv2d(32, 32, 1, stride=1, padding=18)
        self.conv30 = torch.nn.Conv2d(32, 32, 1, stride=1, padding=8)
        self.conv31 = torch.nn.Conv2d(32, 32, 1, stride=1, padding=17)
        self.conv32 = torch.nn.Conv2d(32, 32, 1, stride=1, padding=6)
        self.conv33 = torch.nn.Conv2d(32, 64, 1, stride=1, padding=3)
        self.conv34 = torch.nn.Conv2d(64, 64, 1, stride=1, padding=12)
        self.conv35 = torch.nn.Conv2d(64, 64, 1, stride=1, padding=17)
        self.conv36 = torch.nn.Conv2d(64, 64, 1, stride=1, padding=7)
        self.conv37 = torch.nn.Conv2d(64, 64, 1, stride=1, padding=15)
        self.conv38 = torch.nn.Conv2d(64, 64, 1, stride=1, padding=20)
        self.conv39 = torch.nn.Conv2d(64, 64, 1, stride=1, padding=15)
        self.conv40 = torch.nn.Conv2d(64, 64, 1, stride=1, padding=7)
        self.conv41 = torch.nn.Conv2d(64, 64, 1, stride=1, padding=4)
        self.conv42 = torch.nn.Conv2d(64, 64, 1, stride=1, padding=19)
        self.conv43 = torch.nn.Conv2d(64, 64, 1, stride=1, padding=16)
        self.conv44 = torch.nn.Conv2d(64, 64, 1, stride=1, padding=18)
        self.conv45 = torch.nn.Conv2d(64, 64, 1, stride=1, padding=18)
        self.conv46 = torch.nn.Conv2d(64, 64, 1, stride=1, padding=6)
        self.conv47 = torch.nn.Conv2d(64, 64, 1, stride=1, padding=18)
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
        v135 = v133 * 0.7071067811865476
        v136 = torch.erf(v135)
        v137 = v136 + 1
        v138 = v134 * v137
        v139 = self.conv24(v138)
        v140 = v139 * 0.5
        v141 = v139 * 0.7071067811865476
        v142 = torch.erf(v141)
        v143 = v142 + 1
        v144 = v140 * v143
        v145 = self.conv25(v144)
        v146 = v145 * 0.5
        v147 = v145 * 0.7071067811865476
        v148 = torch.erf(v147)
        v149 = v148 + 1
        v150 = v146 * v149
        v151 = self.conv26(v150)
        v152 = v151 * 0.5
        v153 = v151 * 0.7071067811865476
        v154 = torch.erf(v153)
        v155 = v154 + 1
        v156 = v152 * v155
        v157 = self.conv27(v156)
        v158 = v157 * 0.5
        v159 = v157 * 0.7071067811865476
        v160 = torch.erf(v159)
        v161 = v160 + 1
        v162 = v158 * v161
        v163 = self.conv28(v162)
        v164 = v163 * 0.5
        v165 = v163 * 0.7071067811865476
        v166 = torch.erf(v165)
        v167 = v166 + 1
        v168 = v164 * v167
        v169 = self.conv29(v168)
        v170 = v169 * 0.5
        v171 = v169 * 0.7071067811865476
        v172 = torch.erf(v171)
        v173 = v172 + 1
        v174 = v170 * v173
        v175 = self.conv30(v174)
        v176 = v175 * 0.5
        v177 = v175 * 0.7071067811865476
        v178 = torch.erf(v177)
        v179 = v178 + 1
        v180 = v176 * v179
        v181 = self.conv31(v180)
        v182 = v181 * 0.5
        v183 = v181 * 0.7071067811865476
        v184 = torch.erf(v183)
        v185 = v184 + 1
        v186 = v182 * v185
        v187 = self.conv32(v186)
        v188 = v187 * 0.5
        v189 = v187 * 0.7071067811865476
        v190 = torch.erf(v189)
        v191 = v190 + 1
        v192 = v188 * v191
        v193 = self.conv33(v192)
        v194 = v193 * 0.5
        v195 = v193 * 0.7071067811865476
        v196 = torch.erf(v195)
        v197 = v196 + 1
        v198 = v194 * v197
        v199 = self.conv34(v198)
        v200 = v199 * 0.5
        v201 = v199 * 0.7071067811865476
        v202 = torch.erf(v201)
        v203 = v202 + 1
        v204 = v200 * v203
        v205 = self.conv35(v204)
        v206 = v205 * 0.5
        v207 = v205 * 0.7071067811865476
        v208 = torch.erf(v207)
        v209 = v208 + 1
        v210 = v206 * v209
        v211 = self.conv36(v210)
        v212 = v211 * 0.5
        v213 = v211 * 0.7071067811865476
        v214 = torch.erf(v213)
        v215 = v214 + 1
        v216 = v212 * v215
        v217 = self.conv37(v216)
        v218 = v217 * 0.5
        v219 = v217 * 0.7071067811865476
        v220 = torch.erf(v219)
        v221 = v220 + 1
        v222 = v218 * v221
        v223 = self.conv38(v222)
        v224 = v223 * 0.5
        v225 = v223 * 0.707106781186