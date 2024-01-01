
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(2, 26, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(26, 64, 3, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 1, 5, stride=1, padding=2)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        v7 = self.conv2(v6)
        v8 = v7 * 0.7071067811865476
        v9 = v7 * 0.5
        v10 = torch.erf(v9)
        v11 = v2 * v10
        v12 = v7 * 0.5
        v13 = v7 * 0.7071067811865476
        v14 = torch.erf(v13)
        v15 = v12 * v14
        v16 = self.conv3(v11)
        return v15
# Inputs to the model
x1 = torch.randn(1, 2, 41, 153)
