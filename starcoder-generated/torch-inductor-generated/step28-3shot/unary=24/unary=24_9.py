
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 12, 3, stride=2, padding=2)
        self.conv2 = torch.nn.Conv2d(12, 24, 3, stride=4, padding=0)
        self.conv3 = torch.nn.Conv2d(24, 48, 2, stride=1, padding=1)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = v1 > 0
        v3 = v1 - 3
        m1 = torch.median(v3)
        mean = (m1 + m1) / 2
        sigma = (mean * 0.8) - 3
        v4 = torch.clamp(v3, mean - sigma, mean + sigma)
        v5 = torch.where(v2, v4, v3)
        v6 = torch.sigmoid(v5) + 0.2
        v7 = self.conv2(v6)
        v8 = v7 > 0
        v9 = v7 * 3
        m2 = torch.median(v9)
        mean = (m2 + m2) / 2
        sigma = (mean * 0.6)
        v10 = torch.clamp(v9, mean - sigma, mean + sigma)
        v11 = torch.where(v8, v10, v9)
        v12 = torch.abs(v11) + 0.5
        v13 = torch.tanh(v12)
        v14 = self.conv3(v13)
        v15 = torch.where(v2, v14, v13)
        v16 = torch.asin(v15) + 0.5
        v17 = v16 + v16 # Addition
        v18 = v2 > 0
        v19 = v8 * 3
        v20 = torch.where(v18, v19, v9)
        v21 = torch.tanh(v20)
        v22 = self.conv2(v21)
        v23 = v22 > 0
        v24 = v22 * 2
        m3 = torch.median(v24)
        mean = (m3 + m3) / 2
        sigma = (mean * 0.5) - 5
        v25 = torch.clamp(v24, mean - sigma, mean + sigma)
        v26 = torch.where(v23, v25, v24)
        v27 = v26 <= -2.75
        v28 = v26 * -1 # Negation
        v29 = v27 * v28
        v30 = torch.abs(v29)
        v31 = torch.sigmoid(v30) + 0.5
        v32 = torch.where(v23, v25, v31)
        v33 = v16 > 1.8
        v34 = v16 * 1.8
        v35 = torch.where(v33, v34, v16)
        v36 = self.conv1(v35)
        v37 = v36 > 0
        v38 = v29 - 0.5
        v39 = v37 * 1.5
        v40 = torch.where(v37, v38, v39)
        v41 = torch.abs(v40) * 1.5
        v42 = torch.sigmoid(v41)
        v43 = self.conv2(v42)
        v44 = v43 > 0
        v45 = v22 * -5
        m4 = torch.median(v45)
        mean = (m4 + m4) / 2
        sigma = (mean - 1)
        v46 = torch.clamp(v45, mean - sigma, mean + sigma)
        v47 = v46 * 0.3
        v48 = torch.where(v44, v47, v45)
        v49 = v48 + 1
        v50 = v49 * 0.3
        v51 = v49 > 10
        v52 = v51 * 10 
        v53 = torch.where(v51, v52, v49)
        v54 = v53 * 3
        v55 = self.conv1(v54)
        v56 = v55 > 0
        v57 = v55 - 2
        v58 = v57 * -1
        m5 = torch.median(v58)
        mean = (m5 + m5) / 2
        sigma = (mean * 0.75)
        v59 = torch.clamp(v58, mean - sigma, mean + sigma)
        v60 = v59 * -0.3
        v61 = torch.where(v56, v59, v60)
        v62 = torch.sigmoid(v61) + 0.01
        v63 = v43 > 0
        v64 = v56 * 2
        v65 = torch.where(v63, v64, v56)
        v66 = v65 * 2
        v67 = torch.where(v56, v66, v56)
        v68 = v67 + v67
        m6 = torch.median(v68)
        mean = (m6 + m6) / 2
        sigma = (mean - 0.5)
        v69 = torch.clamp(v68, mean - sigma, mean + sigma)
        v70 = self.conv2(v69)
        v71 = v70 > 0
        v72 = v70 * -2
        m7 = torch.median(v72)
        mean = (m7 + m7) / 2
        sigma = (mean * 0.85) - 2.4
        v73 = torch.clamp(v72, mean - sigma, mean + sigma)
        v74 = torch.where(v71, v73, v72)
        v75 = v74 * 3
        v76 = v72 * -0.05
        v77 = self.conv1(v72)
        v78 = v77 > 0
        m8 = torch.median(v76)
        mean = (m8 + m8) / 2
        sigma = (mean - 0.25)
        v79 = torch.clamp(v76, mean - sigma, mean + sigma)
        v80 = v77 - 0.5
        v81 = v80 * -0.2
        v82 = v79 * v81
        v83 = torch.where(v82, v80, v77)
        v84 = v83 * -0.15
        v85 = v75 + v75
        v86 = v85 * -0.5
        v87 = v85 + v85 # Addition
        v88 = v87 * 1000
        v89 = v78 * -1
        v90 = torch.where(v89, v84, v88)
        v91 = v90 * 0.5
        v92 = v75 + v91
        v93 = v92 + v92
        v94 = v93 * 4
        v95 = v71 * -2
        v96 = torch.where(v71, v95, v72)
        v97 = self.conv1(v96)
        v98 = v97 > 0
        v99 = v76 - 0.1
        v100 = v99 * -3.3
        v101 = torch.where(v98, v100, v99)
        v102 = v101 * 1.4
        v103 = torch.abs(v102)
        v104 = v101 * -2.5
        v105 = v104 + v103
        v106 = v98 * 3.75
        v107 = v106 * -2.5
        v108 = v107 - v105
        v109 = v108 * 73
        v110 = v104 * -2.9
        v111 = v110 + v109
        v112 = v110 * -8
        v113 = v98 * 6
        v114 = v113 * 10
        v115 = v113 * 2
        v116 = v115 + v114
        v117 = v116 * 2.5
        v118 = v98 * 5
        v119 = v118 * -1.25
        v120 = v119 - v117
        v121 = v118 * 3
        v122 = v120 + v121
        v123 = self.conv2(v122)-v122
        v124 = v123 > 0
        v125 = self.conv1(v123)
        v126 = v124 * -2.5
        m9 = torch.median(v126)
        mean = (m9 + m9) / 2
        sigma = (mean - 0.5)
        v127 = torch.clamp(v126, mean - sigma, mean + sigma)
        v128 = v125 <= 1
        v129 = torch.where(v124, v126, v127)
        v130 = v128 * v129
        v131 = v130 * 2
        v132 = v125 - 2
        v133 = v130 + v131
        m10 = torch.median(v133)
        mean = (m10 + m10) / 2
        sigma = (mean + 1.25)
        v134 = torch.clamp(v133, mean - sigma, mean + sigma)
        v135 = v128 * v134
        v136 = v123 <= -2
        v137 = v136 * v135
        v138 = self.conv2(v123)
        v139 = torch.where(v124, v135, v138)
        v140 = self.conv1(v139)
        v141 = v135 * 0.75
        m11 = torch.median(v141)
        mean = (m11 + m11) / 2
        sigma = (mean - 1.5)
        v142 = torch.clamp(v141, mean - sigma, mean + sigma)
        v143 = v140 <= 1.9
        v144 = v140 + v140
        v145 = torch.where(v143, v140, v144)
        v146 = v143 * v145
        v147 = v135 - 0.5
        v148 = v147 + v147
        v149 = v141 + v142
        m12 = torch.median(v149)
        mean = (m12 + m12) / 2
        sigma = (mean + 1.5)
        v150 = torch.clamp(v149, mean - sigma, mean + sigma)
        v151 = torch.where(v140, v146, v150)
        v152 = v151 + v151
        v153 = v152 > 5
        v154 = torch.where(v153, v151, v151)
        v155 = v154 * 0.75
        v156 = v140 + v155
        v157 = v156 + v156
        v158 = v157 > 5
        v159 = v113 * -2
        v160 = torch.where(v158, v159, v156)
        v161 = self.conv1(v160)
        v162 = v51 * -2.5
        m13 = torch.median(v162)
        mean = (m13 + m13) / 2
        sigma = (mean + 1.25)
        v163 = torch.clamp(v162, mean - sigma, mean + sigma)
        v164 = v161 > 2.5
        v165 = v161 * -2.5
        v166 = torch.where(v164, v165, v161)
        v167 = v166 * 0.25
        v168 = v113 * 2
        v169 = v168 * -1.5
        v170 = v167 + v169
        m14 = torch.median(v170)
        mean = (m14 + m14) / 2
        sigma = (mean + 0.5)
        v171 = torch.clamp(v170, mean - sigma, mean + sigma)
        v172 = v171 > 20
        v173 = v118 * -3
        v174 = torch.where(v172, v171, v173)
        v175 = v113 + v174
        v176 = self.conv1(v175)
        v177 = v176 > 0
        v178 = v115 * 500
        v179 = v118 * 375
        v180 = v178 + v179
        v181 = v180 * 1.7
        v182 = v177 * -1.5
        v183 = torch.where(v182, v181, v180)
        v184 = v177 * 2.5
        v185 = torch.abs(v184)
        v186 = self.conv1(v185)
        v187 = v186 <= -1.5
        v188 = v180 - 10
        v189 = v188 * 2
        v190 = torch.where(v187, v188, v189)
        v191 = v190 + v190
        m15 = torch.median(v191)
        mean = (m15 + m15) / 2
        sigma = (mean + 0.375)
        v192 = torch.clamp(v191, mean - sigma, mean + sigma)
        v193 = v190 * v192
        v194 = self.conv2(v192)
        v195 = v177 * 10
        v196 = self.conv1(v195)
        v197 = v175 + v196
        v198 = v194 < 0
        v199 = v198 * 4
        v200 = torch.where(v198, 0.5, v199)
        v201 = v200 * 3
        v202 = v193 + v201
        v203 = v197 + v202
        v204 = self.conv2(v203)
        v205 = v204 > 0
        v206 = torch.exp(v204)
        v207 = v197 * 2
        v208 = v196 * 3
        v209 = v207 + v208
        v210 = v196 + v209
        v211 = v210 + v202
        v212 = v205 * 10
        v213 = v196 + v212
        v214 = self.conv1(v213)
        v215 = v177 * -2.5
        v216 = v214 - 20
        v217 = v216 * 0.25
        v218 = torch.where(v205, v215, v217)
        v219 = v214 * -2
        v220 = v218 + v219
        v221 = v220 > 2.5
        v222 = v221 * 10
        v223 = v220 - 2.5
        v224 = v222 + v223
        v225 = self.conv2(v224)
        v226 = torch.where(v205, v225, v204)
        v227 = self.conv1(v226)
        v228 = v227 > 0
        v229 = v227 * -1
        v230 = self.conv1(v226)
        m16 = torch.median(v230)
        mean = (m16 + m16) / 2
        sigma = (mean + 1)
        v231 = torch.clamp(v230, mean - sigma, mean + sigma)
        v232 = v231 + v231
        v233 = v227 * 20
        v234 = v232 + v233
        m17 = torch.median(v234)
        mean = (m17 + m17) / 2
        sigma = (mean - 2)
        v235 = torch.clamp(v234, mean - sigma, mean + sigma)
        v236 = self.conv1(v235)
        v237 = v236 * 2
        m18 = torch.median(v237)
        mean = (m18 + m18) / 2
        sigma = (mean - 0.75)
        v238 = torch.clamp(v237, mean - sigma, mean + sigma)
        v239 = torch.where(v236 * -1, 200, v238)
        v240 = v235 + v239
        m19 = torch.median(v240)
        mean = (m19 + m19) / 2
        sigma = (mean - 20)
        v241 = torch.clamp(v240, mean - sigma, mean + sigma)
        v242 = self.conv1(v241)
        v243 = self.conv2(v242)
        v244 = v243 < 0
        v245 = v244 * -3.5
        v246 = self.conv1(v241)
        v247 = v235 + v245
        v248 = v244 * 0.1
        m20 = torch.median(v248)
        mean = (m20 + m20) / 2
        sigma = (mean - 2.5)
        v249 = torch.clamp(v248, mean - sigma, mean + sigma)
        v250 = v247 + v249
        v251 = to