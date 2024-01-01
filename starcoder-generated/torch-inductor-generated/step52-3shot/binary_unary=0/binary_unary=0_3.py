
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.relu = torch.nn.ReLU()
    def forward(self, x1, x2, x3):
        v1 = self.conv1(x1)
        v2 = self.conv2(x1)
        v3 = self.conv1(x1)
        v4 = v1 + v2
        v5 = self.relu(v4)
        v6 = x1 + v5
        v7 = self.relu(v6)
        v8 = self.conv3(v7)
        v9 = v8 + x3
        v10 = torch.nn.ReLU()(v9)
        v11 = self.conv2(v10)
        v12 = v11 + x2
        v13 = torch.nn.ReLU()(v12)
        v14 = self.conv3(v13)
        v15 = v14 + x2
        v16 = self.relu(v15)
        return v16
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
x3 = torch.randn(1, 16, 64, 64)
