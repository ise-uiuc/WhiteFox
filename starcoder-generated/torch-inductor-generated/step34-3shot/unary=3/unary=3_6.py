
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(96, 13, 3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(13, 6, 1, stride=2, padding=0)
        self.conv3 = torch.nn.Conv2d(6, 57, 1, stride=1, padding=0)
        self.conv4 = torch.nn.Conv2d(57, 38, 3, stride=2, padding=1)
        self.conv5 = torch.nn.Conv2d(38, 13, 3, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.tanh(v1)
        v3 = self.conv2(v2)
        v4 = torch.abs(v3)
        v5 = torch.sigmoid(v4)
        v6 = self.conv3(v5)
        v7 = v6 * 0.5
        v8 = v6 * 0.7071067811865476
        v9 = torch.erf(v8)
        v10 = v9 + 1
        v11 = v7 * v10
        v12 = self.conv4(v11)
        v13 = v12 * 0.5
        v14 = v12 * 0.7071067811865476
        v15 = torch.erf(v14)
        v16 = v15 + 1
        v17 = v13 * v16
        v18 = self.conv5(v17)
        v19 = v18 * 0.5
        v20 = v18 * 0.7071067811865476
        v21 = torch.erf(v20)
        v22 = v21 + 1
        v23 = v19 * v22
        return v23
# Inputs to the model
x1 = torch.randn(1, 96, 1, 1)
