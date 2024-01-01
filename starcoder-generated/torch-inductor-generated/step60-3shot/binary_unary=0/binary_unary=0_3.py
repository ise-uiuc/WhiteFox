
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x1, x2, x3):
        v1 = self.conv1(x1)
        v2 = v1 + x3
        v3 = torch.relu(v2)
        a1 = self.conv2(v3)
        v4 = self.conv3(v3)
        v5 = self.conv1(v4)
        v6 = v1 + v5
        v7 = torch.relu(v6)
        a2 = self.conv2(v7)
        v8 = self.conv3(v7)
        v9 = self.conv1(v8)
        v10 = v9 + a2
        v11 = torch.relu(v10)
        v12 = self.conv3(v11)
        v13 = self.conv1(v12)
        v14 = v13 + a1
        v15 = torch.relu(v14)
        return v15
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
x3 = torch.randn(1, 16, 64, 64)
