
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 16, 1, stride=1, padding=0)
        self.conv4 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv5 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv6 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x1, x2, x3, x4, x5):
        v1 = self.conv1(x1)
        v2 = v1 + x1
        v3 = torch.relu(v2)
        v4 = self.conv2(v3)
        v5 = v4 + x2
        v6 = torch.relu(v5)
        v7 = self.conv3(x3)
        v8 = v7 + x3
        v9 = torch.relu(v8)
        v10 = self.conv4(v9)
        v11 = v10 + x4
        v12 = torch.relu(v11)
        v13 = self.conv5(v12)
        v14 = v13 + x5
        v15 = torch.relu(v14)
        v16 = v15 + x1
        v17 = torch.relu(v16)
        v18 = self.conv6(v17)
        v19 = self.conv1(v18) + v9
        v20 = torch.relu(v19)
        return v20
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
x3 = torch.randn(1, 16, 64, 64)
x4 = torch.randn(1, 16, 64, 64)
x5 = torch.randn(1, 16, 64, 64)
