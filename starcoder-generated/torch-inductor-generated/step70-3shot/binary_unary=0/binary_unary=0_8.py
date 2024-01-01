
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x1, x2, x3):
        v1 = self.conv1(x1)
        v2 = self.conv2(x1)
        v3 = self.conv1(x2)
        v4 = v1 + v2
        v5 = torch.relu(v4)
        v6 = v1 + v3
        v7 = torch.relu(v6)
        v8 = v5 + v7
        v9 = torch.relu(v8)
        a1 = self.conv3(x3)
        v10 = v9 + a1
        v11 = torch.relu(v10)
        a2 = self.conv3(x1)
        v12 = v11 + a2
        v13 = torch.relu(v12)
        return v13
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
x3 = torch.randn(1, 16, 64, 64)
