
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 8, 3, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(8, 8, 3, stride=1, padding=0)
        self.conv5 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
        self.conv6 = torch.nn.Conv2d(8, 2, 3, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        v7 = self.conv2(v6)
        v8 = self.conv3(v7)
        v9 = v8 * 0.5
        v10 = v8 * 0.7071067811865476
        v11 = torch.erf(v10)
        v12 = v11 + 1
        v13 = v9 * v12
        v14 = self.conv4(v13)
        v15 = self.conv5(v14)
        v16 = v15 * 0.5
        v17 = v15 * 0.7071067811865476
        v18 = torch.erf(v17)
        v19 = v18 + 1
        v20 = v16 * v19
        v21 = self.conv6(v20)
        return v21
# Inputs to the model
x1 = torch.randn(1, 8, 112, 112)
