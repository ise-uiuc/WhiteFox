
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 4, 5, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(4, 6, 5, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(6, 8, 5, stride=1, padding=0)
        self.conv4 = torch.nn.Conv2d(8, 10, 5, stride=1, padding=0)
        self.conv5 = torch.nn.Conv2d(10, 12, 5, stride=1, padding=0)
        self.conv6 = torch.nn.Conv2d(12, 14, 5, stride=1, padding=0)
        self.conv7 = torch.nn.Conv2d(14, 16, 5, stride=1, padding=0)
        self.conv8 = torch.nn.Conv2d(16, 18, 5, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = F.sigmoid(v1)
        v3 = v1 * v2
        v4 = self.conv2(v3)
        v5 = F.sigmoid(v4)
        v6 = v4 * v5
        v7 = self.conv3(v6)
        v8 = F.sigmoid(v7)
        v9 = v7 * v8
        v10 = self.conv4(v9)
        v11 = F.sigmoid(v10)
        v12 = v10 * v11
        v13 = self.conv5(v12)
        v14 = F.sigmoid(v13)
        v15 = v13 * v14
        v16 = self.conv6(v15)
        v17 = F.sigmoid(v16)
        v18 = v16 * v17
        v19 = self.conv7(v18)
        v20 = F.sigmoid(v19)
        v21 = v19 * v20
        v22 = self.conv8(v21)
        v23 = torch.sigmoid(v22)
        v24 = v22 * v23
        return v24
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
