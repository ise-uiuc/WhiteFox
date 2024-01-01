
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 17, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(17, 12, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(12, 7, 1, stride=1, padding=0)
        self.conv4 = torch.nn.Conv2d(7, 17, 3, stride=1, padding=1)
        self.conv5 = torch.nn.Conv2d(17, 9, 1, stride=1, padding=0)
        self.conv6 = torch.nn.Conv2d(9, 10, 3, stride=1, padding=1)
        self.conv7 = torch.nn.Conv2d(10, 4, 1, stride=1, padding=0)
        self.conv8 = torch.nn.Conv2d(4, 13, 3, stride=1, padding=1)
        self.conv9 = torch.nn.Conv2d(13, 17, 1, stride=1, padding=0)
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
        v14 = v13 * 0.5
        v15 = v13 * 0.7071067811865476
        v16 = torch.erf(v15)
        v17 = v16 + 1
        v18 = v14 * v17
        v19 = self.conv4(v18)
        v20 = self.conv5(v19)
        v21 = self.conv6(v20)
        v22 = self.conv7(v21)
        v23 = self.conv8(v22)
        v24 = self.conv9(v23)
        return v24
# Inputs to the model
x1 = torch.randn(1, 1, 256, 256)
