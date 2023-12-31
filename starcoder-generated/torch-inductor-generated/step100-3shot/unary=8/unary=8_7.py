
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(41, 6, 5)
        self.conv2 = torch.nn.Conv2d(6, 8, 5)
        self.conv3 = torch.nn.Conv2d(8, 10, 5)
        self.conv4 = torch.nn.Conv2d(10, 12, 5)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 - v1
        v3 = self.conv2(v2)
        v4 = v3 - v3
        v5 = self.conv3(v4)
        v6 = v5 - v5
        v7 = self.conv4(v6)
        v8 = v7 - v7
        v9 = v1 * v8
        v10 = torch.relu(v9)
        v11 = v10 / v1
        v12 = v8 + v4
        v13 = torch.relu(v12)
        v14 = v13 / v3
        v15 = v4 + v6
        v16 = torch.relu(v15)
        v17 = v16 / v5
        v18 = v6 + v11
        v19 = torch.relu(v18)
        v20 = v19 / v7
        v21 = v11 * v3
        v22 = v21 + v5
        v23 = torch.relu(v22)
        v24 = v23 / v2
        v25 = v13 * v7
        v26 = v25 + v1
        v27 = torch.relu(v26)
        v28 = v27 / v4
        v29 = v16 * v1
        v30 = v29 + v3
        v31 = torch.relu(v30)
        v32 = v31 / v6
        v33 = v18 * v5
        v34 = v33 + v7
        v35 = torch.relu(v34)
        v36 = v35 / v8
        v37 = v20 + v24
        v38 = torch.relu(v37)
        v39 = v38 / v10
        v40 = v24 + v28
        v41 = torch.relu(v40)
        v42 = v41 / v12
        v43 = v28 + v32
        v44 = torch.relu(v43)
        v45 = v44 / v15
        v46 = v32 + v36
        v47 = torch.relu(v46)
        v48 = v47 / v17
        v49 = v36 + v9
        v50 = torch.relu(v49)
        v51 = v50 / v21
        v52 = v9 + v13
        v53 = torch.relu(v52)
        v54 = v53 / v23
        v55 = v13 + v16
        v56 = torch.relu(v55)
        v57 = v56 / v25
        v58 = v16 + v18
        v59 = torch.relu(v58)
        v60 = v59 / v27
        v61 = v18 + v20
        v62 = torch.relu(v61)
        v63 = v62 / v29
        v64 = v22 + v30
        v65 = torch.relu(v64)
        v66 = v65 / v33
        v67 = v30 + v34
        v68 = torch.relu(v67)
        v69 = v68 / v37
        v70 = v34 + v38
        v71 = torch.relu(v70)
        v72 = v71 / v40
        v73 = v38 + v43
        v74 = torch.relu(v73)
        v75 = v74 / v41
        v76 = v43 + v46
        v77 = torch.relu(v76)
        v78 = v77 / v45
        v79 = v46 + v50
        v80 = torch.relu(v79)
        v81 = v80 / v51
        v82 = v50 + v53
        v83 = torch.relu(v82)
        v84 = v83 / v54
        v85 = v53 + v56
        v86 = torch.relu(v85)
        v87 = v86 / v57
        v88 = v56 + v59
        v89 = torch.relu(v88)
        v90 = v89 / v60
        v91 = v59 + v61
        v92 = torch.relu(v91)
        v93 = v92 / v63
        v94 = v23 + v64
        v95 = torch.relu(v94)
        v96 = v95 / v66
        v97 = v64 + v67
        v98 = torch.relu(v97)
        v99 = v98 / v68
        v100 = v67 + v70
        v101 = torch.relu(v100)
        v102 = v101 / v71
        v103 = v70 + v73
        v104 = torch.relu(v103)
        v105 = v104 / v75
        v106 = v73 + v76
        v107 = torch.relu(v106)
        v108 = v107 / v78
        v109 = v76 + v79
        v110 = torch.relu(v109)
        v111 = v110 / v81
        v112 = v79 + v82
        v113 = torch.relu(v112)
        v114 = v113 / v84
        v115 = v82 + v85
        v116 = torch.relu(v115)
        v117 = v116 / v87
        v118 = v85 + v88
        v119 = torch.relu(v118)
        v120 = v119 / v90
        v121 = v88 + v91
        v122 = torch.relu(v121)
        v123 = v122 / v93
        v124 = v94 + v97
        v125 = torch.relu(v124)
        v126 = v125 / v99
        v127 = v97 + v100
        v128 = torch.relu(v127)
        v129 = v128 / v101
        v130 = v100 + v103
        v131 = torch.relu(v130)
        v132 = v131 / v105
        v133 = v103 + v106
        v134 = torch.relu(v133)
        v135 = v134 / v108
        v136 = v106 + v109
        v137 = torch.relu(v136)
        v138 = v137 / v111
        v139 = v109 + v112
        v140 = torch.relu(v139)
        v141 = v140 / v114
        v142 = v112 + v115
        v143 = torch.relu(v142)
        v144 = v143 / v117
        v145 = v115 + v118
        v146 = torch.relu(v145)
        v147 = v146 / v119
        v148 = v118 + v121
        v149 = torch.relu(v148)
        v150 = v149 / v123
        v151 = v124 + v127
        v152 = torch.relu(v151)
        v153 = v152 / v129
        v154 = v25 + v125
        v155 = torch.relu(v154)
        v156 = v155 / v126
        v157 = v12 + v131
        v158 = torch.relu(v157)
        v159 = v158 / v128
        v160 = v13 + v134)
        v161 = torch.relu(v160)
        v162 = v161 / v130)
        v163 = v16)
        v164 = torch.relu(v163)
        v165 = v164 / v132)
        v166 = v15)
        v167 = torch.relu(v166)
        v168 = v167 / v136)
        v169 = v16)
        v170 = torch.relu(v169)
        v171 = v170 / v137)
        v172 = v10 + v141)
        v173 = torch.relu(v172)
        v174 = v173 / v138)
        v175 = v140 + v14)
        v176 = torch.relu(v175)
        v177 = v176 / v141)
        v178 = v14)
        v179 = torch.relu(v178)
        v180 = v179 / v142)
        v181 = v142 + v145)
        v182 = torch.relu(v181)
        v183 = v182 / v143)
        v184 = v154)
        v185 = torch.relu(v184)
        v186 = v185 / v185)
        self.save_to_hub("pattern-model", None, None, "To show how to save the model trained in this notebook", None, True)
# Inputs to the model
x1 = torch.randn(1, 41, 31, 17)
