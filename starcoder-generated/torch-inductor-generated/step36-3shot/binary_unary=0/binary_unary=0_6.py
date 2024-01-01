
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = torch.relu(v1)
        return v2
# Inputs to the model
x = torch.randn(1, 16, 64, 64)
# Model begins

# Model begins
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x6, x2, x5):
        v1 = self.conv2(x2)
        v2 = v1 + x6
        v3 = torch.relu(v2)
        v4 = self.conv1(v3)
        v5 = v4 + x5
        v6 = self.conv3(v5)
        return v6
# Inputs to the model
x6 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
x5 = torch.randn(1, 16, 64, 64)
# Model end

# Model begins
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x6, x2, x5):
        v1 = self.conv2(x2)
        v2 = v1 + x6
        v3 = torch.relu(v2)
        v4 = self.conv1(v3)
        v5 = v4 + x5
        return v5
# Inputs to the model
x6 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
x5 = torch.randn(1, 16, 64, 64)
