
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 100, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(100, 12, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(12, 70, 2, stride=2, padding=0)
        self.conv4 = torch.nn.Conv2d(70, 82, 5, stride=1, padding=2)
        self.conv5 = torch.nn.Conv2d(82, 3, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        v7 = torch.tanh(v6)
        v8 = self.conv2(v7)
        v9 = torch.atanh(v8)
        v10 = self.conv3(v9)
        v11 = v10 * 0.5
        v12 = self.conv4(v11)
        v13 = v12 * 0.7071067811865476
        v14 = torch.erf(v13)
        v15 = v14 + 1
        v16 = v11 * v15
        return v16
# Inputs to the model
x1 = torch.randn(1, 1, 45, 32)
