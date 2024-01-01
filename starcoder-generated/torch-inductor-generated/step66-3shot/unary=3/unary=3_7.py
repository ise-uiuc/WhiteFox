
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 10, 3)
        self.conv2 = torch.nn.Conv2d(10, 30, 5)
        self.conv3 = torch.nn.Conv2d(30, 10, 5)
        self.conv4 = torch.nn.Conv2d(10, 1, 3)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 * 0.139626
        v3 = v1 * 0.279253
        v4 = torch.erf(v3)
        v5 = v4 + 0.841345
        v6 = v2 * v5
        v7 = self.conv2(v6)
        v8 = v7 * 0.139626
        v9 = v7 * 0.279253
        v10 = torch.erf(v9)
        v11 = v10 + 0.841345
        v12 = v8 * v11
        v13 = self.conv3(v12)
        v14 = v13 * 0.139626
        v15 = v13 * 0.279253
        v16 = torch.erf(v15)
        v17 = v16 + 0.841345
        v18 = v14 * v17
        v19 = self.conv4(v18)
        return v19
# Inputs to the model
x1 = torch.randn(1, 1, 1024, 1)
