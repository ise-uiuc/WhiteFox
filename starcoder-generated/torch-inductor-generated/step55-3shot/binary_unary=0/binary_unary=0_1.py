
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1)
    def forward(self, x1, x2, x3, x4):
        v1 = self.conv1(x1)
        v2 = v1 + x1
        v3 = torch.relu(v2)
        v4 = self.conv2(x3)
        v5 = v4 + x3
        v6 = torch.relu(v5)
        a1 = self.conv3(x2)
        a2 = a1 + x2
        a3 = torch.relu(a2)
        a4 = self.conv4(a3)
        a5 = a4 + v3
        a6 = torch.relu(a5)
        v7 = v6 + v1
        v8 = torch.relu(v7)
        v9 = self.conv3(v8)
        v10 = v1 + v9
        v11 = torch.relu(v10)
        return v11
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
x3 = torch.randn(1, 16, 64, 64)
x4 = torch.randn(1, 16, 64, 64)
