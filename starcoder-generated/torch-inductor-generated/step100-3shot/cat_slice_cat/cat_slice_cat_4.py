
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2):
        v1 = torch.cat([x1, x2], dim=1)
        v2 = v1[:, 0:9223372036854775807]
        v3 = v2[:, 0:16]
        v4 = torch.cat([v1, v3], dim=1)
        v5 = v4[:, 16:32]
        v6 = v1[:, 1:17]
        v7 = v6 + v3
        v8 = v7[0]
        v9 = v7[1]
        v10 = v7[2]
        v12 = v7[3]
        v14 = v7[4]
        v16 = v7[5]
        v18 = v7[6]
        v20 = v7[7]
        v22 = v7[8]
        v24 = v7[9]
        v26 = v7[10]
        v28 = v7[11]
        v30 = v7[12]
        v32 = v7[13]
        v34 = v7[14]
        v36 = v7[15]
        v38 = v7[16]
        v40 = v7[17]
        v42 = v7[18]
        v44 = v7[19]
        v46 = v7[20]
        v48 = v7[21]
        v50 = v7[22]
        v52 = v7[23]
        v54 = v7[24]
        v56 = v7[25]
        v58 = v7[26]
        v60 = v7[27]
        v62 = v7[28]
        v64 = v7[29]
        v66 = v7[30]
        v68 = v7[31]
        v70 = v7[32]
        v72 = v7[33]
        v74 = v7[34]
        v76 = v7[35]
        v78 = v7[36]
        v80 = v7[37]
        v82 = v7[38]
        v84 = v7[39]
        v86 = v7[40]
        v88 = v7[41]
        v90 = v7[42]
        v92 = v7[43]
        v94 = v7[44]
        v96 = v7[45]
        v98 = v7[46]
        v100 = v7[47]
        v102 = v7[48]
        v104 = v7[49]
        v106 = v7[50]
        v108 = v7[51]
        v110 = v7[52]
        v112 = v7[53]
        v114 = v7[54]
        v116 = v7[55]
        v118 = v7[56]
        v120 = v7[57]
        v122 = v7[58]
        v124 = v7[59]
        v126 = v7[60]
        v128 = v7[61]
        v130 = v7[62]
        v132 = v7[63]
        v134 = v7[64]
        v136 = v7[65]
        v138 = v7[66]
        v140 = v7[67]
        v142 = v7[68]
        v144 = v7[69]
        v146 = v7[70]
        v148 = v7[71]
        v150 = v7[72]
        v152 = v7[73]
        v154 = v7[74]
        v156 = v7[75]
        v158 = v7[76]
        v160 = v7[77]
        v162 = v7[78]
        v164 = v7[79]
        v166 = v7[80]
        v168 = v7[81]
        v170 = v7[82]
        v172 = v7[83]
        v174 = v7[84]
        v176 = v7[85]
        v178 = v7[86]
        v180 = v7[87]
        v182 = v7[88]
        v184 = v7[89]
        v186 = v7[90]
        v188 = v7[91]
        v190 = v7[92]
        v192 = v7[93]
        v194 = v7[94]
        v196 = v7[95]
        v198 = v7[96]
        v200 = v7[97]
        v202 = v7[98]
        v204 = v7[99]
        v206 = v7[100]
        v208 = v7[101]
        v211 = torch.Tensor.permute(v8)
        return v3[42]

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 256, 17, 17)
x2 = torch.randn(1, 256, 17, 17)
