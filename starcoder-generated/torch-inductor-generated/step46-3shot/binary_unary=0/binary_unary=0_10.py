
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x1, x2, x3):
        v1 = self.conv1(x1)
        v2 = self.conv2(x1)
        v3 = v1 + x2
        v4 = torch.relu(v3)
        v5 = v1 + v4
        v6 = torch.relu(v5)
        v7 = v7 + x3
        v8 = torch.relu(v3)
        v9 = v6 + v8
        v10 = torch.relu(v9)
        v11 = self.conv3(v10)
        v12 = 1 + v11
        v13 = torch.relu(v12)
        return v13
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
x3 = torch.randn(1, 16, 64, 64)
