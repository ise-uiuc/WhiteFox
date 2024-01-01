
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 17, 3, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(17, 17, 1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(17, 2, 1, stride=1, padding=0)
        self.conv4 = torch.nn.Conv2d(2, 2, 1, stride=1, padding=0)
        self.conv5 = torch.nn.Conv2d(2, 5, 1, stride=1, padding=0)
        self.conv6 = torch.nn.Conv2d(5, 8, 1, stride=1, padding=0)
        self.conv7 = torch.nn.Conv2d(8, 5, 1, stride=1, padding=0)
        self.conv8 = torch.nn.Conv2d(5, 2, 1, stride=1, padding=0)
        self.conv9 = torch.nn.Conv2d(2, 17, 1, stride=1, padding=0)
        self.conv10 = torch.nn.Conv2d(17, 1, 1, stride=1, padding=0)
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
        v14 = self.conv4(v13)
        v15 = self.conv5(v14)
        v16 = self.conv6(v15)
        v17 = self.conv7(v16)
        v18 = self.conv8(v17)
        v20 = v9 * 0.5
        v21 = v9 * 0.7071067811865476
        v22 = torch.erf(v21)
        v23 = v22 + 1
        v24 = v20 * v23
        v25 = self.conv9(v24)
        v26 = v25 * 0.5
        v27 = v25 * 0.7071067811865476
        v28 = torch.erf(v27)
        v29 = v28 + 1
        v31 = v26 * v29
        v32 = self.conv10(v31)
        return v32
# Inputs to the model
x1 = torch.randn(1, 1, 231, 239)
