
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv4 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x1, x2, x3):
        v1 = self.conv1(x1)
        v2 = v1 + x2
        v3 = torch.relu(v2)
        v4 = self.conv2(v3)
        v5 = v1 + x2
        v6 = torch.relu(v5)
        v7 = self.conv3(v6)
        v8 = v7 + v2
        v9 = torch.relu(v8)
        v10 = self.conv4(v9)
        v11 = v10 + v7
        v12 = torch.relu(v11)
        v13 = self.conv3(x2)
        v14 = v13 + v10
        v15 = torch.relu(v14)
        return v15
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv4 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x1, x2, x3):
        v1 = self.conv1(x1)
        v2 = v1 + x2
        v3 = torch.relu(v2)
        v4 = self.conv3(x2)
        v5 = v2 + v4
        v6 = torch.relu(v5)
        v7 = self.conv2(v6)
        v8 = v7 + v3
        v9 = torch.relu(v8)
        v10 = self.conv4(v9)
        v11 = v10 + v7
        v12 = torch.relu(v11)
        return v12
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv4 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x1, x2, x3):
        v1 = self.conv1(x1)
        v2 = x3
        v3 = torch.relu(v2)
        v4 = x3 + v2
        v5 = torch.relu(v4)
        v6 = v1 + x2
        v7 = torch.relu(v6)
        v8 = x3 + v6
        v9 = torch.relu(v8)
        v10 = v7 + x1
        v11 = torch.relu(v10)
        v12 = self.conv2(v11)
        v13 = v1 + x2
        v14 = torch.relu(v13)
        v15 = self.conv4(v14)
        v16 = v15 + v5
        v17 = torch.relu(v16)
        return v17
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv4 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x1, x2, x3):
        v1 = self.conv1(x1)
        v2 = x3
        v3 = torch.relu(v2)
        v4 = x3 + v2
        v5 = torch.relu(v4)
        v6 = self.conv2(v5)
        v7 = v6 + v3
        v8 = torch.relu(v7)
        v9 = self.conv3(v8)
        v10 = v2 + v9
        v11 = torch.relu(v10)
        v12 = self.conv4(v11)
        v13 = torch.relu(v7)
        v14 = v13 + v6
        v15 = torch.relu(v14)
        return v15
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv4 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x1, x2, x3):
        v1 = self.conv1(x1)
        v2 = x3
        v3 = torch.relu(v2)
        v4 = self.conv2(v1)
        v5 = x3 + v4
        v6 = torch.relu(v5)
        v7 = self.conv3(v6)
        v8 = v2 + v7
        v9 = torch.relu(v8)
        v10 = self.conv4(v9)
        v11 = v5 + v10
        v12 = torch.relu(v11)
        return v12
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
x3 = torch.randn(1, 16, 64, 64)
