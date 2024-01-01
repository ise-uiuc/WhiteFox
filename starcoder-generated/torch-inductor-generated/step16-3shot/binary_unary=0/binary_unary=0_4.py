
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 2, stride=2, padding=0)
        self.conv2 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1)
    def forward(self, x1, x2, x3, x4):
        v1 = self.conv1(x1)
        v2 = v1 + x2
        v3 = torch.relu(v2)
        v4 = self.conv2(v3)
        v5 = v4 + x3
        v6 = torch.relu(v5)
        v7 = self.conv3(v6)
        v8 = torch.sigmoid(v7)
        v9 = v8.size(1)
        v10 = v9 + x4
        v11 = v10.size(1)
        v12 = v11 + x1
        return v12
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x4 = torch.randn(1, 16, 64, 64)
x2 = -42
x3 = 100
