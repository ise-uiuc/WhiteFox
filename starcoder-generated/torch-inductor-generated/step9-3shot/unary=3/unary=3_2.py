
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 8, 2, stride=2, padding=1)
        self.conv1 = torch.nn.Conv2d(8, 16, 4, stride=1, padding=2)
        self.conv2 = torch.nn.Conv2d(16, 32, 4, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(32, 64, 3, stride=1, padding=2)
        self.conv4 = torch.nn.Conv2d(64, 128, 2, stride=1, padding=0)
        self.conv5 = torch.nn.Conv2d(128, 16, 3, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.tanh(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        v7 = self.conv1(v6)
        v8 = v7 * 0.5
        v9 = v7 * 0.7071067811865476
        v10 = torch.tanh(v9)
        v11 = v10 + 1
        v12 = v8 * v11
        v13 = self.conv2(v12)
        v14 = v13 * 0.5
        v15 = v13 * 0.7071067811865476
        v16 = torch.tanh(v15)
        v17 = v16 + 1
        v18 = v14 * v17
        v19 = self.conv3(v18)
        v20 = self.conv4(v19)
        v21 = v20 * 0.5
        v22 = v20 * 0.7071067811865476
        v23 = torch.tanh(v22)
        v24 = v23 + 1
        v25 = v21 * v24
        v26 = self.conv5(v25)
        return v26
# Inputs to the model
x1 = torch.randn(1, 1, 112, 112)
