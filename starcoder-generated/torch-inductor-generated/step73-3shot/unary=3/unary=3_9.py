
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 6, 4, stride=1, padding=2)
        self.conv2 = torch.nn.Conv2d(6, 16, 4, stride=1, padding=2)
        self.conv3 = torch.nn.Conv2d(16, 64, 4, stride=1, padding=2)
    def forward(self, x):
        x = torch.abs(x)
        v1 = self.conv(x)
        v2 = torch.abs(v1)
        v4 = v2 * 0.5
        v5 = v2 * 0.7071067811865476
        v6 = torch.erf(v5)
        v7 = v6 + 1
        v8 = v4 * v7
        v9 = self.conv2(v8)
        v10 = torch.abs(v9)
        v12 = v10 * 0.5
        v13 = v10 * 0.7071067811865476
        v14 = torch.erf(v13)
        v15 = v14 + 1
        v16 = v12 * v15
        v17 = self.conv3(v16)
        return v17
# Inputs to the model
x1 = torch.randn(1, 3, 33, 47)
