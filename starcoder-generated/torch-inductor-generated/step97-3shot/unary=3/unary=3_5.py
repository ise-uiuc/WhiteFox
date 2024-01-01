
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 973, 13, stride=1, padding=6)
        self.conv2 = torch.nn.Conv2d(63, 955, 13, stride=1, padding=6)
        self.conv3 = torch.nn.Conv2d(63, 945, 13, stride=1, padding=6)
        self.conv4 = torch.nn.Conv2d(63, 934, 13, stride=1, padding=6)
        self.conv5 = torch.nn.Conv2d(973, 32, 7, stride=1, padding=3)
        self.conv6 = torch.nn.Conv2d(955, 16, 7, stride=1, padding=3)
        self.conv7 = torch.nn.Conv2d(945, 8, 7, stride=1, padding=3)
        self.conv8 = torch.nn.Conv2d(934, 4, 7, stride=1, padding=3)
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
        v25 = v6 + v12 + v18 + v24
        v26 = self.conv5(v25)
        v27 = v26 * 0.5
        v28 = v26 * 0.7071067811865476
        v29 = torch.erf(v28)
        v30 = v29 + 1
        v31 = v27 * v30
        v32 = v12 + v18 + v24 + v31
        v33 = self.conv6(v32)
        v34 = v33 * 0.5
        v35 = v33 * 0.7071067811865476
        v36 = torch.erf(v35)
        v37 = v36 + 1
        v38 = v34 * v37
        v39 = v18 + v24 + v31 + v38
        v40 = self.conv7(v39)
        v41 = v40 * 0.5
        v42 = v40 * 0.7071067811865476
        v43 = torch.erf(v42)
        v44 = v43 + 1
        v45 = v41 * v44
        v46 = v24 + v31 + v38 + v45
        v47 = self.conv8(v46)
        return v47
# Inputs to the model
x1 = torch.randn(1, 1, 259, 129)
