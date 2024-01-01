
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(8, 8, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(8, 8, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(8, 8, 7, stride=1, padding=3)
        self.conv4 = torch.nn.Conv2d(8, 8, 7, stride=1, padding=3)
        self.conv5 = torch.nn.Conv2d(8, 8, 7, stride=1, padding=3)
        self.conv6 = torch.nn.Conv2d(8, 8, 7, stride=1, padding=3)
        self.conv7 = torch.nn.Conv2d(8, 8, 7, stride=1, padding=3)
        self.conv8 = torch.nn.Conv2d(8, 8, 7, stride=1, padding=3)
    def forward(self, x1, x2, x3, x4, x5):
        v1 = self.conv1(x1)
        v2 = torch.relu(x2 + v1)
        v3 = self.conv2(v2)
        v4 = v3 + x1
        v5 = torch.relu(x2 + v4)
        v6 = self.conv3(v5)
        v7 = v6 + x2
        v8 = torch.relu(x1 + v7)
        v9 = v5 + x3
        v10 = self.conv4(v8)
        v11 = torch.relu(x2 + v9)
        v12 = v10 + x4
        v13 = torch.relu(x3 + v1)
        v14 = self.conv5(v13)
        v15 = v14 + x5
        v16 = torch.relu(v10 + v15)
        v17 = v2 + x3
        v18 = v12 + x2
        v19 = v1 + x4
        v20 = self.conv6(v12)
        v21 = v20 + x2
        v22 = v18 + v16
        v23 = v21 + v2
        v24 = self.conv7(v23)
        v25 = torch.relu(v24)
        v26 = self.conv8(v25)
        v27 = self.conv5(x1)
        v28 = v26 + v16
        v29 = v1 + x5
        return v28
# Inputs to the model
x1 = torch.randn(1, 8, 64, 64)
x2 = torch.randn(1, 8, 64, 64)
x3 = torch.randn(1, 8, 64, 64)
x4 = torch.randn(1, 8, 64, 64)
x5 = torch.randn(1, 8, 64, 64)
