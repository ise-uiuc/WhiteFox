
x1 = torch.randn(1, 1, 8, 8)
x2 = torch.randn(1, 3, 4, 4)
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.relu = torch.nn.ReLU()
        self.avg_pool = torch.nn.AvgPool2d(4, stride = 2, padding = 0)
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(48,60)
        self.fc2 = nn.Linear(60,48)
        self.fc3 = nn.Linear(48,48)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x1, x2):
        v0 = self.flatten(x2.detach())
        x1 = self.conv1(x1)
        x1 = self.relu(x1)
        x2 = self.conv2(x2)
        x2 = self.relu(x2)
        x3 = x1 + x2
        v1 = self.relu(x3)
        v2 = self.avg_pool(v1)
        v3 = self.adaptive_avg_pool(v1)
        v4 = v2 + v3
        v5 = self.flatten(v4)
        v6 = self.flatten(v4)
        v7 = v5 + v6
        v8 = self.fc1(v7)
        v8 = self.sigmoid(v8)
        v9 = self.fc2(v7)
        v9 = v8.mul(v9)
        v11 = self.fc2(v9.detach())
        v12 = self.fc3(v11)
        x4 = v12 + v12
        x5 = x4.mul(x4)
        v13 = self.flatten(x4)
        v14 = self.flatten(x4)
        v15 = v13 + v14
        v16 = self.fc1(v15)
        v16 = v9.mul(v16)
        x6 = self.fc3(v9)
        x7 = v16 + x6
        x8 = self.flatten(x7)
        x9 = self.fc1(x8)
        x10 = self.fc2(x8)
        x11 = self.fc3(x8)
        x12 = x9 + x10
        x13 = self.sigmoid(x12)
        x14 = x11 + x11
        x15 = x12 + x14
        x16 = x13.detach()
        x17 = x15.detach()
        x18 = x16 + x17
        x19 = x15.mul(x18)
        v17 = x19.detach()
        return v17
