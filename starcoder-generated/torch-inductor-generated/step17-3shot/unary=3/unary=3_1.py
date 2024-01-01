
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(32, 32, 3, stride=1, padding=0, dilation=2)
        self.conv2 = torch.nn.Conv2d(32, 32, 1, stride=1, padding=0, dilation=2)
        self.conv3 = torch.nn.Conv2d(32, 32, 3, stride=1, padding=2, dilation=2)
        self.conv4 = torch.nn.Conv2d(32, 32, 3, stride=1, padding=2, dilation=2)
        self.conv5 = torch.nn.Conv2d(32, 32, 3, stride=1, padding=1, dilation=2)
        self.conv6 = torch.nn.Conv2d(32, 32, 1, stride=1, padding=0, dilation=2)
    def forward(self, x1):
        v1 = self.conv1(x1)
        b1 = nn.BatchNorm2d(32)
        v2 = b1(v1)
        v3 = v2 * 0.5
        v4 = v2 * 0.7071067811865476
        v5 = torch.erf(v4)
        v6 = v5 + 1
        v7 = v3 * v6
        v8 = self.conv2(v1)
        v9 = self.conv3(v1)
        v10 = self.conv4(v1)
        v11 = self.conv5(v1)
        v12 = self.conv6(v3)
        v13 = v11 * 0.5
        v14 = v11 * 0.7071067811865476
        v15 = torch.erf(v14)
        v16 = v15 + 1
        v17 = v13 * v16
        v18 = (v7 - v17) * v12
        v19 = v8 * v18
        v20 = b1(v3)
        v21 = b1(v8)
        v22 = b1(v9)
        v23 = b1(v10)
        return v19
# Inputs to the model
x1 = torch.randn(1, 32, 90, 90)
