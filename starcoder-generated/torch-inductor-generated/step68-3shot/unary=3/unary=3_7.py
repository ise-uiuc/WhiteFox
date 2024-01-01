
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(5, 10, 3, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(10, 25, 3, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(25, 30, 1, stride=3, padding=0)
        self.conv4 = torch.nn.Conv2d(30, 100, 1, stride=4, padding=0)
        self.conv5 = torch.nn.Conv2d(100, 120, 1, stride=2, padding=0)
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
        v19 = self.conv3(v12)
        v13 = torch.erf(v19)
        v14 = v13 + 1
        v15 = v12 * v14
        v16 = self.conv4(v19)
        v17 = torch.erf(v16)
        v18 = v17 + 1
        v20 = v16 * v18
        return self.conv5(v20)
# Inputs to the model
x1 = torch.randn(1, 5, 58, 58)
