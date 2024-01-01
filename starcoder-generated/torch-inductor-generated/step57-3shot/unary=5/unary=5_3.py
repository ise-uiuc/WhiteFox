
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose1d(32, 67, 7, stride=12, padding=3)
        self.t438 = torch.tensor([True, False], dtype=torch.bool)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        if self.t438[0]:
            v228 = 1e-09
        else:
            v228 = 1e-14
        v7 = v1 * v228
        v8 = torch.abs(v7)
        v9 = v8[0][0]
        v10 = torch.tensor([-2.22044605e-16, 3.12256033e-01, 2.19883915e-01, 1.91627034e-01, 1.64721749e-01, 1.39598757e-01, 1.16755615e-01, 9.67076657e-02, 7.99912606e-02, 6.62372535e-02, 5.50299019e-02, 4.60014927e-02, 3.88306242e-02, 3.32333287e-02, 2.89595177e-02, 2.57898200e-02, 2.35282562e-02, 2.20007463e-02, 2.10503731e-02], dtype=torch.float32)
        v11 = v9 < v10[0]
        if v11[0]:
            v12 = v10[12]
        else:
            v13 = v1[0][2][0]
            v12 = v13 * v13
        if v11[1]:
            v14 = v10[13]
        else:
            v15 = v1[0][3][2]
            v14 = v15 * v15
        if v11[2]:
            v16 = v10[14]
        else:
            v17 = v1[0][9][0]
            v16 = v17 * v17
        if v11[3]:
            v18 = v10[15]
        else:
            v19 = v1[0][17][3]
            v18 = v19 * v19
        if v11[4]:
            v20 = v10[16]
        else:
            v21 = v1[0][5][1]
            v20 = v21 * v21
        if v11[5]:
            v22 = v10[17]
        else:
            v23 = v1[0][1][2]
            v22 = v23 * v23
        if v11[6]:
            v24 = v10[18]
        else:
            v25 = v1[0][2][3]
            v24 = v25 * v25
        if v11[7]:
            v26 = v10[19]
        else:
            v27 = v1[0][6][3]
            v26 = v27 * v27
        if v11[8]:
            v28 = v10[20]
        else:
            v29 = v1[0][15][1]
            v28 = v29 * v29
        if v11[9]:
            v30 = v10[21]
        else:
            v31 = v1[0][4][3]
            v30 = v31 * v31
        if v11[10]:
            v32 = v10[22]
        else:
            v33 = v1[0][0][2]
            v32 = v33 * v33
        if v11[11]:
            v34 = v10[23]
        else:
            v35 = v1[0][12][0]
            v34 = v35 * v35
        if v11[12]:
            v36 = v10[24]
        else:
            v37 = v1[0][13][1]
            v36 = v37 * v37
        if v11[13]:
            v38 = v10[25]
        else:
            v39 = v1[0][4][0]
            v38 = v39 * v39
        if v11[14]:
            v40 = v10[26]
        else:
            v41 = v1[0][18][3]
            v40 = v41 * v41
        if v11[15]:
            v42 = v10[27]
        else:
            v43 = v1[0][16][0]
            v42 = v43 * v43
        v44 = v12 + v14 + v16 + v18 + v20 + v22 + v24 + v26 + v28 + v30 + v32 + v34 + v36 + v38 + v40 + v42
        if self.t438[1]:
            v45 = 1e-09
        else:
            v45 = 1e-14
        v46 = torch.tensor([v2, v44], dtype=torch.float32) * v45
        return v46
# Inputs to the model
x1 = torch.randn(1, 32, 3)
