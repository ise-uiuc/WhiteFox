
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 4, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(1, 3, 2, stride=1, padding=1)
        self.conv3 = torch.nn.ConvTranspose2d(3, 3, 2, stride=2, padding=0)
        self.conv4 = torch.nn.Conv2d(3, 1, 3, stride=2, padding=1)
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
        v13 = self.conv3(v12)
        v14 = v7 * 1.4142135623730951
        v15 = v12 * 1.4142135623730951
        v16 = v10 * 1.4142135623730951
        v17 = v16 + 1
        v18 = v14 * v17
        v19 = self.conv4(v18)
        v20 = v2 * 1.4142135623730951
        v21 = v17 * 1.4142135623730951
        return v19
# Inputs to the model
x1 = torch.randn(1, 1, 18, 18)
