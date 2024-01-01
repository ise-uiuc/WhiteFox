
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(47, 26, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(26, 48, 5, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(48, 77, 3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(77, 31, 5, stride=1, padding=0)
        self.conv5 = torch.nn.Conv2d(31, 35, 3, stride=1, padding=1)
        self.conv6 = torch.nn.Conv2d(35, 19, 5, stride=1, padding=0)
        self.gap2 = torch.nn.AdaptiveAvgPool2d(7)
        self.gap3 = torch.nn.AdaptiveAvgPool2d(1)
        self.linear1 = torch.nn.Linear(100, 59)
        self.linear2 = torch.nn.Linear(59, 32)
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
        v25 = self.conv5(v24)
        v26 = v25 * 0.5
        v27 = v25 * 0.7071067811865476
        v28 = torch.erf(v27)
        v29 = v28 + 1
        v30 = v26 * v29
        v31 = self.conv6(v30)
        v32 = self.gap2(v31)
        v33 = self.gap3(v31)
        v34 = v32.view(-1, 7)
        v35 = self.linear1(v34)
        v36 = v35 * 0.5
        v37 = v35 * 0.7071067811865476
        v38 = torch.erf(v37)
        v39 = v38 + 1
        v40 = v36 * v39
        v41 = self.linear2(v40)
        return v41
# Inputs to the model
x1 = torch.randn(1, 47, 26, 26)
