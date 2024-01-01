
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv4 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv5 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x1, x2, x3, x4, x5, x6, x7, x8):
        v1 = self.conv1(x1)
        v2 = v1 + x2
        v3 = torch.relu(v2)
        v4 = v3 + x2
        v5 = torch.relu(v4)
        v6 = self.conv2(v5)
        v7 = v6 + x3
        v8 = torch.relu(v7)
        v9 = v8 + x3
        v10 = torch.relu(v9)
        v11 = self.conv3(v10)
        v12 = v11 + x4
        v13 = torch.relu(v12)
        v14 = self.conv4(v4)
        v15 = 1 + v14
        v16 = v15 + x5
        v17 = torch.relu(v16)
        v18 = self.conv5(v17)
        v19 = v18 + x6
        v20 = torch.relu(v19)
        v21 = torch.relu(v20)
        v22 = v21 + x7
        v23 = torch.relu(v22)
        v24 = self.conv6(v23)
        v25 = v24 + x8
        return v25
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
x3 = torch.randn(1, 16, 64, 64)
x4 = torch.randn(1, 16, 64, 64)
x5 = torch.randn(1, 16, 64, 64)
x6 = torch.randn(1, 16, 64, 64)
x7 = torch.randn(1, 16, 64, 64)
x8 = torch.randn(1, 16, 64, 64)
