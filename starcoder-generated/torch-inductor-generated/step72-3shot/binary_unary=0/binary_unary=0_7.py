
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv4 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x1, x2, x3):
        v1 = self.conv1(x1)
        v2 = self.conv1(x2)
        v3 = v1 * v2
        v4 = torch.relu(v3)
        v5 = v3 + x2
        v6 = torch.relu(v5)
        v7 = self.conv2(v6)
        v8 = self.conv2(x3)
        v9 = 2*v7 + v8
        v10 = torch.relu(v9)
        v11 = self.conv3(v10)
        v12 = v11 * v7
        v13 = torch.relu(v12)
        v14 = self.conv4(v13)
        v15 = v14 * v5
        v16 = torch.relu(v15)
        return v16
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
x3 = torch.randn(1, 16, 64, 64)
