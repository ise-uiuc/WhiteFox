
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 8, 2, stride=2, padding=1) # Using stride=2
        self.conv1 = torch.nn.Conv2d(8, 16, 4, stride=1, padding=2)
        self.conv2 = torch.nn.Conv2d(16, 32, 4, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(32, 64, 3, stride=1, padding=2)
        self.conv4 = torch.nn.Conv2d(64, 128, 2, stride=1, padding=0) # Using stride=1, padding=0 for a square kernel
        self.conv5 = torch.nn.Conv2d(128, 128, 3, stride=1, padding=0) # Using stride=1, padding=0 for non-square kernels
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        v7 = self.conv1(v6)
        v8 = v7 * 0.5
        v9 = v7 * 0.7071067811865476
        v10 = torch.erf(v9)
        v11 = v10 + 1
        v12 = v8 * v11
        v13 = self.conv2(v12)
        v14 = self.conv3(v13)
        v15 = v14 * 0.5
        v16 = v14 * 0.7071067811865476
        v17 = torch.erf(v16)
        v18 = v17 + 1
        v19 = v15 * v18
        v20 = self.conv4(v19)
        v21 = self.conv5(v20)
        return v21
# Inputs to the model
x1 = torch.randn(1, 1, 112, 112)
