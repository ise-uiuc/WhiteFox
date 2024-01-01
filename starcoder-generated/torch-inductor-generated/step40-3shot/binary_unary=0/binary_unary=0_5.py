
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.bn1  = torch.nn.BatchNorm2d(16)
        self.conv2 = torch.nn.Conv2d(16, 64, 5, stride=2, padding=2)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.bn1(v1)
        v3 = torch.relu(v2)
        v4 = self.conv2(v3)
        v5 = torch.relu(v4)
        return v5
# Inputs to the model
x = torch.randn(1, 16, 64, 64)
# model ends

# Model begins
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(16)
        self.conv2 = torch.nn.Conv2d(16, 16, 3, stride=2, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(16)
        self.conv3 = torch.nn.Conv2d(16, 16, 3, stride=2, padding=1)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.bn1(v1)
        v3 = torch.relu(v2)
        v4 = self.conv2(v3)
        v5 = self.bn2(v4)
        v6 = torch.relu(v5)
        v7 = self.conv3(v6)
        v8 = torch.relu(v7)
        return v8
# Inputs to the model
x = torch.randn(1, 16, 64, 64)
# model ends

# Model begins
# model begins
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(16)
        self.conv2 = torch.nn.Conv2d(16, 16, 3, stride=2, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(16)
        self.conv3 = torch.nn.Conv2d(16, 16, 3, stride=2, padding=1)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.bn1(v1)
        v3 = torch.relu(v2)
        v4 = self.conv2(v3)
        v5 = self.bn2(v4)
        v6 = torch.relu(v5)
        v7 = self.conv3(v6)
        v8 = torch.relu(v7)
        return v8
# Inputs to the model
x = torch.randn(1, 16, 64, 64)
# model ends
# model begins