
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1a = torch.nn.Conv2d(128, 24, 1, stride=1, padding=0)
        self.conv1b = torch.nn.Conv2d(24, 32, 5, stride=1, padding=2)
        self.conv2a = torch.nn.Conv2d(24, 32, 1, stride=1, padding=0)
        self.conv2b = torch.nn.Conv2d(32, 32, 5, stride=1, padding=2)
        self.conv3a = torch.nn.Conv2d(32, 24, 1, stride=1, padding=0)
        self.conv3b = torch.nn.Conv2d(24, 24, 3, stride=1, padding=0)
        self.conv4a = torch.nn.Conv2d(24, 16, 1, stride=1, padding=0)
        self.conv4b = torch.nn.Conv2d(16, 16, 3, stride=1, padding=0)
    def forward(self, x1):
        v1a = self.conv1a(x1)
        v1b = self.conv1b(v1a)
        v2a = torch.relu(v1b)
        v2b = self.conv2a(v2a)
        v2c = torch.relu(v2b)
        v3a = self.conv2b(v2c)
        v3b = self.conv3a(v3a)
        v3c = torch.relu(v3b)
        v3d = self.conv3b(v3c)
        v4a = torch.relu(v3d)
        v4b = self.conv4a(v4a)
        v4a = torch.relu(v4b)
        v4c = self.conv4b(v4a)
        return v4b
# Inputs to the model
x1 = torch.randn(1, 128, 32, 32)
