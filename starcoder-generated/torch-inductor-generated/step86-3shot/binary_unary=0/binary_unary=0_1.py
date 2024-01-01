
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3, groups=8)
        self.conv2 = torch.nn.Conv2d(16, 16, 5, stride=1, padding=3, groups=16)
        self.conv3 = torch.nn.Conv2d(16, 16, 5, stride=1, padding=3, groups=16)
        self.conv4 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=3, groups=16)
        self.conv5 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1, groups=16)
    def forward(self, x1, x2, x3):
        v1 = self.conv1(x1)
        v2 = v1 + x2
        v3 = torch.relu(v2)
        v4 = self.conv2(v3)
        v5 = v4 + x3
        v6 = self.conv3(v5)
        v7 = v6 - v5
        v8 = self.conv4(v7 - v3)
        v9 = self.conv5(v8)
        v10 = torch.relu(v9)
        v11 = self.conv2(v10)
        v12 = v11 + x3
        v13 = self.conv3(v12)
        v14 = v13 - v12
        v15 = self.conv4(v14 - x1)
        v16 = self.conv2(v15)
        v17 = torch.relu(v16)
        v18 = self.conv2(v17)
        v19 = self.conv4(v18)
        v20 = torch.relu(v19)
        v21 = self.conv5(v20)
        v22 = torch.relu(v21)
        return v22

# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
x3 = torch.randn(1, 16, 64, 64)
