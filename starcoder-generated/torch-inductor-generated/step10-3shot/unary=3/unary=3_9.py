
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 1, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(1, 1, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(1, 1, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(1, 1, 3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(1, 1, 3, stride=1, padding=1)
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
        v22 = v19 * 0.7071067811865476
        v23 = torch.erf(v22)
        v24 = v23 + 1
        v25 = v20*v24
        v26 = self.conv5(v25)
        return v26
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
