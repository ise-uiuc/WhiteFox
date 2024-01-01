
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv3 = torch.nn.Conv2d(29, 1, 1, stride=1, padding=0)
        self.conv5 = torch.nn.Conv2d(1, 50, 3, stride=1, padding=1)
        self.conv7 = torch.nn.Conv2d(50, 197, 7, stride=2, padding=0)
        self.conv9 = torch.nn.Conv2d(197, 158, 3, stride=1, padding=1)
        self.conv11 = torch.nn.Conv2d(158, 159, 7, stride=1, padding=3)
        self.conv13 = torch.nn.Conv2d(159, 50, 1, stride=1, padding=0)
        self.conv15 = torch.nn.Conv2d(50, 176, 5, stride=1, padding=2)
        self.conv17 = torch.nn.Conv2d(176, 40, 1, stride=1, padding=0)
        self.conv19 = torch.nn.Conv2d(40, 172, 3, stride=1, padding=1)
        self.conv21 = torch.nn.Conv2d(172, 36, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv3(x1)
        v2 = self.conv5(v1)
        v3 = v2 * 0.5
        v4 = v2 * 0.7071067811865476
        v5 = torch.erf(v4)
        v6 = v5 + 1
        v7 = v3 * v6
        v8 = self.conv7(v7)
        v9 = v8 * 0.5
        v10 = v8 * 0.7071067811865476
        v11 = torch.erf(v10)
        v12 = v11 + 1
        v13 = v9 * v12
        v14 = self.conv9(v13)
        v15 = v14 * 0.5
        v16 = v14 * 0.7071067811865476
        v17 = torch.erf(v16)
        v18 = v17 + 1
        v19 = v15 * v18
        v20 = self.conv11(v19)
        v21 = v20 * 0.5
        v22 = v20 * 0.7071067811865476
        v23 = torch.erf(v22)
        v24 = v23 + 1
        v25 = v21 * v24
        v26 = self.conv13(v25)
        v27 = v26 * 0.5
        v28 = v26 * 0.7071067811865476
        v29 = torch.erf(v28)
        v30 = v29 + 1
        v31 = v27 * v30
        v32 = self.conv15(v31)
        v33 = v32 * 0.5
        v34 = v32 * 0.7071067811865476
        v35 = torch.erf(v34)
        v36 = v35 + 1
        v37 = v33 * v36
        v38 = self.conv17(v37)
        v39 = v38 * 0.5
        v40 = v38 * 0.7071067811865476
        v41 = torch.erf(v40)
        v42 = v41 + 1
        v43 = v39 * v42
        v44 = self.conv19(v43)
        v45 = v44 * 0.5
        v46 = v44 * 0.7071067811865476
        v47 = torch.erf(v46)
        v48 = v47 + 1
        v49 = v45 * v48
        v50 = self.conv21(v49)
        return v50
# Inputs to the model
x1 = torch.randn(1, 29, 43, 44)
