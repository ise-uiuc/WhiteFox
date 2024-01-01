
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, [3, 1], stride=1, padding=(1, 0))
        self.conv3 = torch.nn.Conv2d(32, 64, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 + v1
        v3 = self.conv2(v2)
        v4 = self.conv2(v2)
        v5 = v3 + v4
        v6 = torch.relu(v5)
        v7 = self.conv3(v6)
        v8 = self.conv3(v6)
        v9 = self.conv3(v6)
        v10 = v7 + v8 + v9
        v11 = torch.relu(v10)
        return v11