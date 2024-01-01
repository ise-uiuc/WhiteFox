
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv4 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x1, x2, x3, x4, t1, t2):
        v1 = self.conv1(x1)
        v2 = v1 + x1
        v3 = torch.relu(v2)
        v4 = self.conv2(v3)
        v5 = v4 + x1
        v6 = torch.relu(v5)
        v7 = self.conv3(v6)
        v8 = v7 + v2
        v9 = torch.relu(v8)
        v10 = self.conv4(v9)
        v11 = torch.relu(t1 + v10)
        v12 = torch.relu(x4 + v11 + t2)
        v13 = torch.cat([[1], [2], [3]], axis=0)
        v14 = v13 * v12
        v15 = torch.cat([t1, x2, v14], axis=0)
        v16 = x3 + v15
        return v16
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
x3 = torch.randn(1, 16, 64, 64)
x4 = torch.randn(1, 16, 64, 64)
t1 = torch.randn(1, 16, 64, 64)
t2 = torch.randn(1, 16, 64, 64)
