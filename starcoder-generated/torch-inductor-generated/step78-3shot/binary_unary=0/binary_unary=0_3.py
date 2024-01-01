
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2_1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2_2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv4 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv5 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3_1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3_2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3_3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv6 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x1, x2, x3, x4, x5, x6, x7, x8):
        v1 = self.conv1(x1)
        v2 = v1 + x2
        v3 = torch.relu(v2)
        v4 = self.conv2(v3)
        v5 = v4 + x3
        v6 = torch.relu(v5)
        v7 = self.conv2_1(v6)
        v8 = v7 + x4
        v9 = torch.relu(v8)
        v10 = self.conv2_2(v9)
        v11 = v10 + v2
        v12 = torch.relu(v11)
        v13 = self.conv4(v12)
        v14 = v13 + x5
        v15 = torch.relu(v14)
        v16 = self.conv5(v15)
        v17 = v16 + x6
        v18 = torch.relu(v17)
        v19 = self.conv3_1(v18)
        v20 = v19 + x7
        v21 = torch.relu(v20)
        v22 = self.conv3_2(v21)
        v23 = v22 + x8
        v24 = torch.relu(v23)
        v25 = self.conv3_3(v24)
        v26 = v25 + v13
        v27 = torch.relu(v26)
        v28 = self.conv6(v27)
        return v28
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
x3 = torch.randn(1, 16, 64, 64)
x4 = torch.randn(1, 16, 64, 64)
x5 = torch.randn(1, 16, 64, 64)
x6 = torch.randn(1, 16, 64, 64)
x7 = torch.randn(1, 16, 64, 64)
x8 = torch.randn(1, 16, 64, 64)
