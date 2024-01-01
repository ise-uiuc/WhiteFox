
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 207, 1, stride=1, padding=0)
        self.conv2 = torch.nn.ConvTranspose2d(207, 208, 1, stride=1, padding=0)
        self.conv3 = torch.nn.ConvTranspose2d(208, 193, 21, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.8323361595199321
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
        return v13
# Inputs to the model
x1 = torch.randn(1, 1, 60, 34)
