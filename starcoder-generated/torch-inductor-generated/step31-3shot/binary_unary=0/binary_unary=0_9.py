
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv4 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.conv5 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.conv6 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.conv7 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.conv8 = torch.nn.Conv2d(16, 16, 1, stride=1, padding=0)
    def forward(self, x1, x2, x3):
        v1 = self.conv1(x3)
        v2 = v1 + x2
        v3 = torch.relu(v2)
        v4 = self.conv2(v3)
        v5 = v4 * 2
        v6 = v3 + x2
        v7 = x1 + v6
        v8 = self.conv3(v7)
        v9 = v8 * v5
        v10 = v8 + v5
        v11 = self.conv4(v6)
        v12 = self.conv5(v9)
        v13 = self.conv6(v10)
        v14 = self.conv7(v12)
        v15 = v14 * v11
        v16 = v15 * v13
        v17 = v15 * v16
        v18 = self.conv8(v17)
        v19 = v18 * v11
        v20 = v19 + v11
        return v20
# Inputs to the model
x1 = torch.randn(50, 16, 32, 32)
x2 = torch.randn(50, 16, 32, 32)
x3 = torch.randn(50, 16, 32, 32)
