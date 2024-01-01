
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(2, 38, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(38, 29, 5, stride=1, padding=2)
        self.conv3 = torch.nn.Conv2d(29, 23, 1, stride=1, padding=0)
        self.conv4 = torch.nn.Conv2d(23, 2, 1, stride=1, padding=0)
        self.conv5 = torch.nn.Conv2d(2, 37, 2, stride=1, padding=0)
        self.conv6 = torch.nn.Conv2d(37, 4, 1, stride=1, padding=0)
        self.conv7 = torch.nn.Conv2d(4, 5, 1, stride=1, padding=0)
        self.conv8 = torch.nn.Conv2d(5, 4, 1, stride=1, padding=0)
        self.conv9 = torch.nn.Conv2d(4, 3, 1, stride=1, padding=0)
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
        v25 = torch.abs(v24)
        v26 = self.conv5(v25)
        v27 = v26 * 0.5
        v28 = v26 * 0.7071067811865476
        v29 = torch.erf(v28)
        v30 = v29 + 1
        v31 = v27 * v30
        v32 = self.conv6(v31)
        v33 = v32 * 0.5
        v34 = v32 * 0.7071067811865476
        v35 = torch.erf(v34)
        v36 = v35 + 1
        v37 = v33 * v36
        v38 = self.conv7(v37)
        v39 = v38 * 0.5
        v40 = v38 * 0.7071067811865476
        v41 = torch.erf(v40)
        v42 = v41 + 1
        v43 = v39 * v42
        v44 = self.conv8(v43)
        v45 = v44 * 0.5
        v46 = v44 * 0.7071067811865476
        v47 = torch.erf(v46)
        v48 = v47 + 1
        v49 = v45 * v48
        v50 = self.conv9(v49)
        return v50

# Inputs to the model
x1 = torch.randn(1, 2, 32, 32)
