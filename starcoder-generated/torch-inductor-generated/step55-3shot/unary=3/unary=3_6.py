
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 64, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(32, 2, 3, stride=1, padding=1)
        self.conv4 = torch.nn.ConvTranspose2d(3, 64, 3, stride=1, padding=1)
        self.conv5 = torch.nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1)
        self.conv6 = torch.nn.ConvTranspose2d(32, 2, 3, stride=1, padding=1)
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
        res12 = v13
        v14 = self.conv4(res12)
        res11 = v14 + x1
        v15 = self.conv5(res11)
        res10 = v15
        v16 = self.conv6(res10)
        return v16
# Inputs to the model
x1 = torch.randn(1, 3, 45, 45)
