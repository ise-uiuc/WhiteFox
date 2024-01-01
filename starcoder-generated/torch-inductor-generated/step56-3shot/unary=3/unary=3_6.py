
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 64, 3, stride=1, padding=2, dilation=2)
        self.conv4 = torch.nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv5 = torch.nn.Conv2d(128, 64, 3, stride=1, padding=3, dilation=2)
        self.conv6 = torch.nn.Conv2d(64, 64, 3, stride=2, padding=3, dilation=2)
        self.conv7 = torch.nn.Conv2d(64, 1, 3, stride=1, padding=2, dilation=2)
        self.convt7 = torch.nn.ConvTranspose2d(1,1,3,stride=1, padding=2, dilation=2)
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
        
        v43 = self.convt7(v42)
        return v43


# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
