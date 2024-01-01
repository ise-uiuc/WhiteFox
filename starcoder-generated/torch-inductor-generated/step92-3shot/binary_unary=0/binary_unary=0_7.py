
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(16, 16, 1, stride=1, padding=0)
        self.conv4 = torch.nn.Conv2d(16, 16, 1, stride=1, padding=0)
    def forward(self, x1, x2, x3):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = v2 + x3
        v4 = torch.relu(v3)
        v5 = self.conv1(v4)
        v6 = self.conv3(v5)
        v7 = v6 + v2
        v8 = torch.relu(v7)
        v9 = self.conv3(v8)
        v10 = v9 + x2
        v11 = torch.relu(v10)
        v12 = self.conv4(v11)
        v13 = v12 + x2
        v14 = torch.relu(v13)
        v15 = self.conv4(v14)
        v16 = v15 + x2
        v17 = torch.relu(v16)
        v18 = self.conv4(v17)
        v19 = v18 + x2
        v20 = torch.relu(v19)
        v21 = self.conv1(v20)
        v22 = self.conv2(v21)
        v23 = v22 + v14
        v24 = torch.relu(v23)
        return v24
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
x3 = torch.randn(1, 16, 64, 64)
