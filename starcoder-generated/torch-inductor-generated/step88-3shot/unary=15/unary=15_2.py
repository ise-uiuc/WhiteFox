
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1a = torch.nn.Conv2d(1, 8, 1, stride=1, padding=0)
        self.conv1b = torch.nn.Conv2d(1, 64, 1, stride=1, padding=0)
        self.conv2a = torch.nn.Conv2d(8, 4, 1, stride=1, padding=0)
        self.conv2b = torch.nn.Conv2d(8, 4, 1, stride=1, padding=0)
        self.conv3a = torch.nn.Conv2d(16, 4, 1, stride=1, padding=0)
        self.conv3b = torch.nn.Conv2d(16, 32, 1, stride=1, padding=0)
        self.conv4a = torch.nn.Conv2d(64, 32, 1, stride=1, padding=0)
        self.conv4b = torch.nn.Conv2d(64, 32, 1, stride=1, padding=0)
        self.conv5 = torch.nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv6 = torch.nn.Conv2d(64, 128, 1, stride=1, padding=0)
        self.conv7 = torch.nn.Conv2d(256, 128, 3, stride=2, padding=1)
        self.conv8 = torch.nn.Conv2d(256, 64, 3, stride=2, padding=1)
    def forward(self, x1):
        v1a = self.conv1a(x1)
        v1b = self.conv1b(x1)
        v1c = torch.tanh(v1a)
        v1d = torch.tanh(v1b)
        v1e = v1c + v1d
        v2a = self.conv2a(v1e)
        v2b = self.conv2b(v1e)
        v2c = torch.tanh(v2a)
        v2d = torch.tanh(v2b)
        v3a = self.conv3a(v2c + v2d)
        v3b = self.conv3b(v2c + v2d)
        v4a = self.conv4a(v1c + v1d)
        v4b = self.conv4b(v1c + v1d)
        v4c = torch.tanh(v4a)
        v4d = torch.tanh(v4b)
        v5 = torch.sigmoid(v3a + v3b + v4c + v4d)
        return v4c + v4d
# Inputs to the model
x1 = torch.randn(1, 1, 32, 56)
