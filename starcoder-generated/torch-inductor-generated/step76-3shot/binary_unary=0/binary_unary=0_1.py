
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x1, x2, x3, x4):
        v1 = self.conv2(x1)
        a1 = self.conv2(x4)
        v2 = v1 + x1
        v3 = torch.relu(v2)
        v4 = self.conv3(x4)
        v5 = self.conv2(v3) + v4
        v6 = torch.relu(v5)
        v7 = self.conv3(v6) + x2
        v8 = torch.relu(v7)
        v9 = self.conv1(v8)  # <-- NEW: This is the only difference
        v10 = torch.relu(v9)
        v11 = self.conv3(v10) + x3
        v12 = torch.relu(v11)
        return v12
