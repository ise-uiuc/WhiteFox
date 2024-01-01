
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(5, 21, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(21, 12, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(12, 16, 3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(16, 23, 3, stride=2, padding=1)
        self.conv5 = torch.nn.Conv2d(23, 29, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        v7 = self.conv2(v6)
        v8 = self.conv3(v7)
        v9 = self.conv4(v8)
        v10 = v9 * 0.5
        v11 = v9 * 0.7071067811865476
        v12 = torch.erf(v11)
        v13 = v12 + 1
        v14 = v10 * v13
        v15 = self.conv5(v14)
        v16 = self.conv4(v15)
        v17 = v16 * 0.5
        v18 = v16 * 0.7071067811865476
        v19 = torch.erf(v18)
        v20 = v19 + 1
        v21 = v17 * v20
        v22 = self.conv3(v21)
        return v22
# Inputs to the model
x1 = torch.randn(1, 5, 5, 5)
