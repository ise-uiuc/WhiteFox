
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x1, x2, x3):
        v1 = self.conv1(x1)
        v2 = self.conv1(x1)
        v3 = v1 + v2
        v4 = torch.relu(v3)
        v5 = v1 + x2
        a1 = self.conv1(x1)
        a2 = self.conv2(x2)
        v6 = self.conv2(v5)
        v7 = self.conv3(v5)
        v8 = v7 + a1
        v9 = torch.relu(v8)
        v10 = self.conv3(v9)
        v11 = v1 + a2
        v12 = torch.relu(v11)
        v13 = self.conv3(v12)
        v14 = v13 + x3
        v15 = torch.relu(v14)
        return v9, v13
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
x3 = torch.randn(1, 16, 64, 64)
