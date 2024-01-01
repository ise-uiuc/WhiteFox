
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(64, 32, 1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(32, 64, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.pow(v1, 0.25)
        v3 = torch.pow(v2, 4)
        v4 = v3 * 0.5
        v5 = v3 * 0.7071067811865476
        v6 = torch.erf(v5)
        v7 = v6 + 1
        v8 = v4 * v7
        v9 = self.conv2(v8)
        v10 = v9 * 0.5
        v11 = v9 * 0.7071067811865476
        v12 = torch.erf(v11)
        v13 = v12 + 1
        v14 = v10 * v13
        v15 = self.conv3(v14)
        return v15
# Inputs to the model
x1 = torch.randn(1, 3, 56, 56)
