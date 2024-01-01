
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 7, 7, stride=3, padding=5)
        self.conv2 = torch.nn.Conv2d(7, 8, (9, 10), stride=(0, 1), padding=(7, 5), dilation=(7, 1))
    def forward(self, x1):
        v1 = self.conv(x1)
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
        return v12
# Inputs to the model
x1 = torch.randn(71, 3, 334)
y1 = torch.randn(53, 58)
z1 = torch.randn(80, 12, 63)
w1 = torch.randn(100, 35, 69)
