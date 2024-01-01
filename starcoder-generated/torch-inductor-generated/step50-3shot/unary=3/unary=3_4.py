
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv22 = torch.nn.Conv2d(1, 1, 1, stride=2, padding=0)
        self.conv24 = torch.nn.Conv2d(1, 1, 7, stride=1, padding=3)
        self.conv26 = torch.nn.Conv2d(1, 1, 5, stride=1, padding=2)
        self.conv28 = torch.nn.Conv2d(1, 1, 1, stride=2, padding=0)
    def forward(self, x1):
        v1 = self.conv22(x1)
        v2 = self.conv24(v1)
        v3 = v2 * 0.5
        v4 = v2 * 0.7071067811865476
        v5 = torch.erf(v4)
        v6 = v5 + 1
        v7 = v3 * v6
        v8 = self.conv26(v7)
        v9 = v8 * 0.5
        v10 = v8 * 0.7071067811865476
        v11 = torch.erf(v10)
        v12 = v11 + 1
        v13 = v9 * v12
        v14 = self.conv28(v13)
        return v14
# Inputs to the model
x1 = torch.randn(1, 1, 61, 61)
