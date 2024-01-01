
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 5, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(16, 32, 1, stride=1)
        self.conv5 = torch.nn.Conv2d(32, 16, 1, stride=1)
        self.conv6 = torch.nn.Conv2d(16, 1, 5, stride=1, padding=2)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = v1 + 4
        v3 = torch.relu(v2)
        v4 = self.conv2(v3)
        v5 = v4 * x2
        v6 = torch.relu(v5)
        v7 = self.conv3(v6)
        v8 = v7 + 10
        v9 = torch.relu(v8)
        v10 = self.conv4(v9)
        v11 = v10 - 4
        v12 = self.conv5(v11)
        v13 = v12 - v9
        v14 = torch.relu(v13)
        v15 = self.conv6(v14)
        v16 = v15 - 2.5
        return v16
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
