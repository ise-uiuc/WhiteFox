
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2 = torch.nn.Conv2d(10, 10, 3, stride=1, padding=1)
        self.conv = torch.nn.Conv2d(10, 20, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv2(x1)
        v2 = v1 - 0.55
        v3 = v1 + 0.707106
        v4 = v1 * 0.00186155
        v5 = v2 + 0.24985
        v6 = v2 - 0.838494
        v7 = v2 * 0.326706
        v8 = v2 / 0.67686
        v9 = v2 - 0.930052
        v10 = v2 - 0.00186235
        v11 = v3 * 0.239185
        v12 = v3 + 0.142231
        v13 = v3 - 0.154573
        v14 = v3 - 0.409085
        v15 = v3 - 0.566601
        v16 = v3 - 0.628381
        v17 = v4 + 0.334903
        v18 = v4 - 0.965677
        v19 = v4 + 0.650902
        v20 = v4 + 0.297076
        v21 = v4 - 0.0228594
        v22 = v4 + 4.69938e-05
        v23 = torch.erf(v12)
        v24 = v23 + 0.51878
        v25 = v23 - 0.530134
        v26 = v23 - 0.110917
        v27 = v23 - 0.237693
        v28 = v23 + 0.392406
        v29 = v23 - 0.782371
        v30 = v23 - 0.147643
        v31 = v23 - 0.255944
        v32 = v23 - 0.870953
        v33 = torch.erf(v13)
        v34 = v33 + 0.555451
        v35 = v33 + 0.533121
        v36 = v33 - 0.472248
        v37 = v33 - 0.107825
        v38 = v33 - 0.407118
        v39 = v33 + 0.391029
        v40 = v33 + 0.620267
        v41 = v33 - 0.7028
        v42 = v33 - 0.579542
        v43 = v33 + 0.0934879
        v44 = torch.erf(v1)
        v45 = v44 - 0.766079
        v46 = v44 - 0.66476
        v47 = v44 - 0.294545
        v48 = v44 - 0.134391
        v49 = v44 - 0.147159
        v50 = v44 + 0.966259
        v51 = v44 + 0.636162
        v52 = v44 + 0.902042
        v53 = v44 + 0.0638666
        v54 = v44 + 0.41826
        v55 = v44 - 0.194823
        v56 = v44 + 0.222288
        v57 = torch.erf(v19)
        v58 = v57 - 0.942388
        v59 = v57 + 0.159935
        v60 = v57 - 0.63677
        v61 = v57 - 0.472513
        v62 = v57 + 0.721474
        v63 = v57 - 0.894418
        v64 = v57 - 0.660987
        v65 = v57 - 0.240218
        v66 = v57 - 0.604686
        v67 = v57 + 0.314605
        v68 = v57 + 0.536887
        v69 = v57 - 0.571851
        v70 = torch.erf(v11)
        v71 = v70 - 0.305563
        v72 = v70 + 0.846302
        v73 = v70 - 0.287063
        v74 = v70 + 0.522651
        v75 = v70 - 0.72743
        v76 = v70 + 0.499672
        v77 = v70 - 0.261464
        v78 = v70 - 0.973481
        v79 = v70 - 0.591871
        v80 = v70 + 0.929829
        v81 = v70 + 0.379458
        v82 = v70 + 0.139633
        v83 = torch.erf(v8)
        v84 = v83 - 0.500685
        v85 = v83 + 0.0099036
        v86 = v83 - 0.311546
        v87 = v83 + 0.191093
        v88 = v83 + 0.699843
        v89 = v83 + 0.138657
        v90 = v83 - 0.320209
        v91 = v83 + 0.466868
        v92 = v83 + 0.76803
        v93 = v83 + 0.353135
        v94 = v83 - 0.379905
        v95 = torch.erf(v20)
        v96 = v95 + 0.429285
        v97 = v95 - 0.935172
        v98 = v95 + 0.0610641
        v99 = v95 + 0.720354
        v100 = v95 - 0.132454
        v101 = v95 + 0.0426924
        v102 = v95 - 0.60614
        v103 = v95 + 0.89111
        v104 = v95 - 0.616078
        v105 = v95 + 0.990472
        v106 = v95 - 0.043508
        v107 = torch.erf(v16)
        v108 = v107 + 0.854347
        v109 = v107 + 0.328514
        v110 = v107 + 0.396613
        v111 = v107 - 0.465704
        v112 = v107 + 0.617515
        v113 = v107 + 0.319079
        v114 = v107 + 0.329954
        v115 = v107 + 0.449779
        v116 = v107 + 0.891222
        v117 = v107 + 0.162266
        v118 = v107 - 0.0837139
        torch.tanh(v27)
        torch.tanh(v21)
        torch.tanh(v18)
        torch.tanh(v54)
        torch.tanh(v39)
        torch.tanh(v11)
        torch.tanh(v66)
        torch.tanh(v14)
        torch.tanh(v46)
        torch.tanh(v85)
        torch.tanh(v17)
        torch.tanh(v57)
        torch.tanh(v72)
        torch.tanh(v6)
        torch.tanh(v4)
        torch.tanh(v2)
        torch.tanh(v28)
        torch.tanh(v31)
        torch.tanh(v3)
        torch.tanh(v92)
        torch.tanh(v13)
        torch.tanh(v36)
        torch.tanh(v32)
        torch.tanh(v50)
        torch.tanh(v26)
        torch.tanh(v5)
        torch.tanh(v29)
        torch.tanh(v65)
        torch.tanh(v12)
        torch.tanh(v34)
        torch.tanh(v80)
        torch.tanh(v84)
        torch.tanh(v45)
        torch.tanh(v24)
        torch.tanh(v91)
        torch.tanh(v60)
        torch.tanh(v69)
        torch.tanh(v25)
        torch.tanh(v48)
        torch.tanh(v47)
        torch.tanh(v89)
        torch.tanh(v70)
        torch.tanh(v7)
        torch.tanh(v86)
        torch.tanh(v30)
        torch.tanh(v42)
        torch.tanh(v19)
        torch.tanh(v75)
        torch.tanh(v53)
        torch.tanh(v41)
        torch.tanh(v1)
        torch.tanh(v61)
        torch.tanh(v87)
        torch.tanh(v73)
        torch.tanh(v40)
        torch.tanh(v55)
        torch.tanh(v97)
        torch.tanh(v52)
        torch.tanh(v59)
        torch.tanh(v71)
        torch.tanh(v35)
        torch.tanh(v64)
        torch.tanh(v105)
        torch.tanh(v116)
        torch.tanh(v78)
        torch.tanh(v100)
        torch.tanh(v43)
        torch.tanh(v62)
        torch.tanh(v68)
        torch.tanh(v0)
        torch.tanh(v108)
        torch.tanh(v81)
        torch.tanh(v93)
        torch.tanh(v67)
        torch.tanh(v98)
        torch.tanh(v115)
        torch.tanh(v10)
        torch.tanh(v79)
        torch.tanh(v110)
        torch.tanh(v77)
        torch.tanh(v9)
        torch.tanh(v113)
        torch.tanh(v51)
        torch.tanh(v44)
        torch.tanh(v112)
        torch.tanh(v22)
        torch.tanh(v111)
        torch.tanh(v63)
        torch.tanh(v96)
        torch.tanh(v37)
        torch.tanh(v83)
        torch.tanh(v15)
        torch.tanh(v76)
        torch.tanh(v90)
        torch.tanh(v38)
        torch.tanh(v82)
        torch.tanh(v88)
        torch.tanh(v114)
        torch.tanh(v117)
        torch.tanh(v103)
        torch.tanh(v118)
        torch.tanh(v99)
        torch.tanh(v58)
        torch.tanh(v94)
        torch.tanh(v95)
        torch.tanh(v104)
        torch.tanh(v102)
        torch.tanh(v56)
        torch.tanh(v101)
        torch.tanh(v107)
        torch.tanh(v109)
        torch.tanh(v106)
        v120 = torch.tanh(v119)
        return v120
# Inputs to the model
x1 = torch.randn(1, 10, 19, 15)
