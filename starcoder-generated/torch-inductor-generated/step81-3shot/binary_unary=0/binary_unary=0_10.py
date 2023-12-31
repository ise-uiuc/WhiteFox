
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(128, 128, 1, stride=1, padding=0)
        self.conv1 = torch.nn.Conv2d(128, 128, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(128, 128, 1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(128, 128, 1, stride=1, padding=0)
        self.conv4 = torch.nn.Conv2d(128, 128, 1, stride=1, padding=0)
        self.conv5 = torch.nn.Conv2d(128, 128, 1, stride=1, padding=0)
        self.conv6 = torch.nn.Conv2d(128, 128, 1, stride=1, padding=0)
        self.conv7 = torch.nn.Conv2d(128, 128, 1, stride=1, padding=0)
        self.conv8 = torch.nn.Conv2d(128, 128, 1, stride=1, padding=0)
        self.conv9 = torch.nn.Conv2d(128, 128, 1, stride=1, padding=0)
        self.conv10 = torch.nn.Conv2d(128, 128, 1, stride=1, padding=0)
        self.conv11 = torch.nn.Conv2d(128, 128, 1, stride=1, padding=0)
        self.conv12 = torch.nn.Conv2d(128, 16, 1, stride=1, padding=0)
    def forward(self, x1, x2, x3):
        v1 = self.conv0(x1)
        v2 = self.conv1(x1)
        v3 = self.conv2(x1)
        v4 = self.conv3(x1)
        v5 = self.conv4(v4)
        v6 = self.conv5(v3 + v5)
        a1 = torch.tanh(v4)
        v7 = self.conv6(v2 + a1)
        a2 = torch.log(v2)
        v8 = self.conv7(v2 + a2)
        v9 = self.conv8(v6 + v7)
        v10 = self.conv9(v9 + v1)
        a3 = torch.sigmoid(a1)
        v11 = self.conv10(v1)
        v12 = self.conv11(v6 + v10)
        a4 = torch.tanh(v12)
        v13 = self.conv12(a4 + x3)
        a5 = torch.tanh(v11)
        v14 = torch.relu(v13 + a5)
        v15 = torch.sigmoid(a5)
        v16 = torch.sigmoid(a3)
        v17 = torch.tanh(x3)
        v18 = torch.tanh(a4)
        v19 = torch.log(v16 + torch.sqrt(v14 + x2 * v17))
        a6 = torch.tanh(v9)
        v20 = torch.tanh(v19)
        v21 = torch.relu(v20)
        v22 = torch.log(v21)
        v23 = torch.sin(v15)
        v24 = torch.sin(v16)
        v25 = torch.relu(v23)
        v26 = torch.relu(v24)
        v27 = torch.cos(v26)
        v28 = torch.sin(v24)
        v29 = torch.cos(v26)
        v30 = torch.tanh(v22)
        v31 = torch.tanh(v21)
        v32 = torch.cos(v15)
        v33 = torch.cos(v15 + v8)
        v34 = torch.cos(v19 + v18)
        v35 = torch.cos(v17)
        v36 = torch.tanh(v27)
        v37 = torch.cos(v35)
        v38 = torch.cos(v28)
        v39 = torch.cos(v32 + v34 + a3)
        v40 = torch.cos(v17 + a4)
        v41 = torch.sin(v19)
        v42 = torch.tanh(v32)
        v43 = torch.sin(v15 + v8)
        v44 = torch.tanh(v36)
        v45 = torch.cos(v37)
        v46 = torch.cos(v33)
        v47 = torch.sin(v38)
        v48 = torch.cos(v39 + v44)
        v49 = torch.sin(v33)
        v50 = torch.tanh(v45)
        v51 = torch.cos(v40)
        v52 = torch.cos(v35 + v42)
        v53 = torch.sin(v41)
        v54 = torch.cos(v39)
        v55 = torch.sin(v46)
        v56 = torch.cos(v43)
        v57 = torch.cos(v52 + v53 + v48)
        v58 = torch.log(v57)
        v59 = torch.tanh(v51)
        v60 = torch.sin(v41 + v52)
        v61 = torch.sin(v44)
        v62 = torch.cos(v32)
        v63 = torch.sin(v40 + v54)
        v64 = torch.tanh(v49)
        v65 = torch.cos(v50 + v56)
        v66 = torch.cos(v40 + v55)
        v67 = torch.cos(v48)
        a7 = torch.cos(v38)
        v68 = torch.cos(v59)
        v69 = torch.sin(v48)
        v70 = torch.tanh(v56)
        v71 = torch.sin(v37 + v70)
        v72 = torch.cos(v55)
        v73 = torch.sin(v60 + a7)
        v74 = torch.cos(v59)
        v75 = torch.sin(v58)
        v76 = torch.sin(v52)
        v77 = torch.cos(0.0)
        v78 = torch.log(v76)
        v79 = torch.sin(v55)
        v80 = torch.cos(v60 + v61)
        v81 = torch.sin(v60 + v63)
        v82 = torch.cos(torch.tanh(v63))
        v83 = torch.cos(v64 + v80)
        v84 = torch.sin(v55 + v63)
        v85 = torch.sin(v67 + v82)
        v86 = torch.cos(v76 + v75 + v71)
        v87 = torch.cos(v68)
        v88 = torch.sin(v63)
        v89 = torch.cos(v59)
        v90 = torch.tanh(v70)
        v91 = torch.cos(v60 + v72)
        v92 = torch.cos(v71)
        v93 = torch.sin(v89)
        v94 = torch.cos(v83)
        v95 = torch.sin(v92)
        v96 = torch.sin(v73 + v91 + v87)
        v97 = torch.sigmoid(v96)
        v98 = torch.cos(v93 + v94)
        v99 = torch.tanh(v91)
        v100 = torch.cos(v90)
        v101 = torch.sin(v60 + v64 + v99)
        v102 = torch.cos(v42)
        a8 = torch.cos(v74)
        v103 = torch.tanh(v98)
        v104 = torch.cos(v73)
        v105 = torch.cos(v92 + a8)
        v106 = torch.cos(v90 + v98)
        v107 = torch.tanh(v69)
        v108 = torch.tanh(v78)
        v109 = torch.cos(v88)
        v110 = torch.cos(v90 + v106)
        v111 = torch.cos(v102)
        v112 = torch.sin(v62 + v69)
        v113 = torch.cos(v66 + v73)
        v114 = torch.cos(v84 + v108)
        v115 = torch.log(v86)
        v116 = torch.log(v67)
        a9 = torch.cos(v85)
        v117 = torch.cos(v80 + a9)
        v118 = torch.sin(v81 + v81)
        v119 = torch.tanh(v85)
        v120 = torch.tanh(v77 + v88)
        v121 = torch.tanh(v118 + v85)
        v122 = torch.log(v83)
        v123 = torch.sin(v68 + v81 + v83 + v89)
        v124 = torch.sin(v79)
        v125 = torch.cos(v65)
        v126 = torch.sin(v78)
        v127 = torch.sin(v124)
        v128 = torch.cos(v53 + v126)
        v129 = torch.sin(v101)
        v130 = torch.cos(v116)
        v131 = torch.sin(v51 + v72)
        v132 = torch.sin(v130 + v113)
        v133 = torch.tanh(v101)
        v134 = torch.tanh(v122)
        v135 = torch.tanh(v110)
        v136 = torch.sin(v68 + v90 + v105)
        v137 = torch.cos(v60 + v93 + v112 + v124)
        v138 = torch.cos(v132 + 0.0)
        v139 = torch.tan(v133)
        v140 = torch.tanh(v69 + v107)
        v141 = torch.cos(v105 + v125)
        v142 = torch.sin(v117)
        v143 = torch.cos(v113 + v125)
        v144 = torch.sin(v117 + v119)
        v145 = torch.sin(v127 + v125)
        v146 = torch.cos(v127)
        v147 = torch.sin(v136)
        v148 = torch.tanh(v113)
        v149 = torch.log(v130 + v127)
        v150 = torch.tan(v107)
        v151 = torch.tan(v85)
        v152 = torch.cos(v127)
        v153 = torch.tanh(v138)
        v154 = torch.cos(v141)
        v155 = torch.cos(v135)
        v156 = torch.cos(v143 + v142)
        v157 = torch.sin(v131)
        v158 = torch.cos(v119 + v145)
        v159 = torch.cos(v128)
        v160 = torch.sin(v95 + v137)
        v161 = torch.cos(v138)
        v162 = torch.cos(v140 + v161)
        v163 = torch.log(v137 + v157)
        v164 = torch.tanh(v131 + v140)
        v165 = torch.sin(v151)
        v166 = torch.sin(v155)
        v167 = torch.cos(v154)
        v168 = torch.cos(v156)
        v169 = torch.log(v141 + v131)
        v170 = torch.cos(v148)
        v171 = torch.cos(v140)
        v172 = torch.sin(v149)
        v173 = torch.cos(v150 + v153)
        v174 = torch.log(v139)
        a10 = torch.cos(v159)
        v175 = torch.log(v170)
        v176 = torch.sin(v152 + v160)
        v177 = torch.sin(v164)
        v178 = torch.cos(v134)
        v179 = torch.sin(v162)
        v180 = torch.sin(v173 + v175)
        v181 = torch.sin(v180)
        v182 = torch.sigmoid(v181)
        v183 = torch.sin(v170 + v168 + v165)
        v184 = torch.cos(v174)
        v185 = torch.tanh(v178)
        v186 = torch.sin(v177)
        v187 = torch.sin(v160)
        v188 = torch.cos(v173)
        v189 = torch.sin(v165 + v170 + v171)
        v190 = torch.tan(v163 + v176)
        v191 = torch.sin(v183 + v139)
        v192 = torch.sin(v186 + v177 + v164)
        v193 = torch.cos(v136)
        v194 = torch.tanh(v146)
        v195 = torch.cos(v149)
        v196 = torch.log(v148 + v155)
        v197 = torch.cos(v167)
        v198 = torch.sin(v163 + v176)
        v199 = torch.sin(v174)
        v200 = torch.cos(v181)
        v201 = torch.sin(v199)
        a11 = torch.sin(v188)
        v202 = torch.tanh(v199 + v134)
        v203 = torch.sin(v201 + v194 + v135)
        a12 = torch.cos(v197)
        v204 = torch.log(v192 + v200)
        v205 = torch.tanh(v204)
        v206 = torch.sin(v185 + v186 + v205 + v187)
        v207 = torch.cos(v190)
        v208 = torch.cos(v195 + v186)
        v209 = torch.cos(v146 + v147 + v196)
        v210 = torch.cos(v192 + v200)
        v211 = torch.cos(v202 + v198 + v198 + v199)
        v212 = torch.cos(v206)
        v213 = torch.tanh(v210)
        v214 = torch.sin(v212)
        v215 = torch.cos(v207)
        v216 = torch.cos(v201)
        v217 = torch.tanh(v193 + v211)
        v218 = torch.cos(v211 + v204)
        v219 = torch.log(v176)
        v220 = torch.log(v150)
        v221 = torch.tanh(v211)
        v222 = torch.log(v209)
        v223 = torch.sin(v158)
        v224 = torch.tanh(v202)
        v225 = torch.tanh(v215 + v158)
        v226 = torch.tanh(v221)
        v227 = torch.tanh(v182)
        v228 = torch.tanh(v214 + v226)
        v229 = torch.cos(v217)
        v230 = torch.cos(v218)
        v231 = torch.sin(v220 + v221)
        v232 = torch.tanh(v218 + v211 + v222)
        v233 = torch.log(v221)
        v234 = torch.sin(v229)
        v235 = torch.log(v203)
        v236 = torch.cos(v209)
        v237 = torch.cos(2.0)
        v238 = torch.tan(v223 + v219 + v222 + torch.tanh(v227))
        v239 = torch.log(v237 + v216 + a11)
        v240 = torch.log(v229 + v236)
        v241 = torch.sin(v238 + v239)
        v242 = torch.cos(v230)
        v243 = torch.cos(v224)
        v244 = torch.sin(v238 + v223)
        v245 = torch.sin(v233 + v225)
        v246 = torch.log(v245)
        v247 = torch.cos(v244)
        v248 = torch.tanh(v240 + v234)
        v249 = torch.tanh(v241)
        v250 = torch.cos(v241)
        v251 = torch.tanh(v242)
        v252 = torch.sin(v235)
        v253 = torch.tanh(v249)
        v254 = torch.tanh(v232 + v241)
        v255 = torch.log(v248)
        v256 = torch.tan(v231 + v242)
        v257 = torch.sigmoid(v256)
     