
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, 3, stride=3, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, stride=3, padding=1)
        self.conv3 = torch.nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(64, 1, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 - 2
        v3 = F.relu(v2)
        v4 = self.conv2(v3)
        v5 = v4 - 4
        v6 = F.relu(v5)
        v7 = self.conv3(v6)
        v8 = v7 - 8
        v9 = F.relu(v8)
        v10 = self.conv4(v9)
        v11 = v10 - 16
        v12 = F.relu(v11)
        return v12
# Inputs to the model
x1 = torch.randn(1, 1, 28, 28)
# Model end

# Model begins
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(10, 16, 3, stride=2, padding=2)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, stride=2, padding=2)
        self.flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(4608, 128)
        self.fc2 = torch.nn.Linear(128, 2)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 - 17
        v3 = self.conv2(v2)
        v4 = v3 - 34
        v5 = self.flatten(v4)
        v6 = self.fc1(v5)
        v7 = v6 - 68
        v8 = self.fc2(v7)
        v9 = F.relu(v8)
        return v9
# Inputs to the model
x1 = torch.randn(1, 10, 109, 109)
