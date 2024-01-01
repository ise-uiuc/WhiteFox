
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 1, 1, stride=2, padding=0)
        self.conv2 = torch.nn.Conv2d(32, 16, 1, stride=2, padding=0)
        self.conv3 = torch.nn.Conv2d(32, 1, 5, stride=1, padding=2)
        self.conv4 = torch.nn.Conv2d(1, 1001, 2, stride=2, padding=0)
        self.conv5 = torch.nn.ConvTranspose2d(1001, 338, 2, stride=2, padding=0, output_padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        v7 = self.conv2(v6)
        v5 = v7 + v1
        v8 = v7 * 0.5
        v9 = v7 * 0.7071067811865476
        v10 = torch.erf(v9)
        v11 = v10 + 1
        v12 = v8 * v11
        v13 = v5 * v12
        v4 = self.conv3(v13)
        v14 = v4 + v5
        v6 = v14 * 0.5
        v7 = v14 * 0.7071067811865476
        v8 = torch.erf(v7)
        v9 = v7 * 0.7071067811865476
        v10 = torch.erf(v7)
        v11 = v10 + 1
        v12 = v6 * v11)
        v13 = v14 * 0.5
        v14 = v14 * 0.7071067811865476
        v15 = torch.erf(v14)
        v16 = v15 + 1
        v17 = v13 * v16
        v18 = self.conv4(v17)
        v19 = v18 * 0.5
        v20 = v18 * 0.7071067811865476
        v21 = torch.erf(v20)
        v22 = v21 + 1
        v23 = v19 * v22
        v24 = self.conv5(v23)
        return v24
# Inputs to the model
x1 = torch.randn(1, 16, 128, 119)
