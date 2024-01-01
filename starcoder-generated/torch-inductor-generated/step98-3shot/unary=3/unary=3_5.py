
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(10, 20, 5, stride=1, padding=2)
        self.conv2 = torch.nn.ConvTranspose2d(20, 20, 4, stride=2, padding=1)
        self.conv3 = torch.nn.ConvTranspose2d(20, 30, 3, stride=2, padding=0)
        self.conv4 = torch.nn.Conv2d(30, 9, 7, stride=1, padding=2)
        self.conv5 = torch.nn.Conv2d(9, 5, 1, stride=1, padding=0)
        self.conv6 = torch.nn.Conv2d(5, 1, 1, stride=1, padding=0)
        self.conv7 = torch.nn.Conv2d(10, 20, 5, stride=1, padding=2)
        self.conv8 = torch.nn.Conv2d(20, 1, 7, stride=1, padding=2)
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
        v32 = self.conv7(v31)
        v33 = v32 * 0.5
        v34 = v32 * 0.7071067811865476
        v35 = torch.erf(v34)
        v36 = v35 + 1
        v37 = v33 * v36
        v38 = self.conv8(v37)
        return v38
# Inputs to the model
x1 = torch.randn(1, 10, 39, 39)
