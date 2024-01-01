
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 24, 3, stride=1, padding=1)
        self.conv2a = torch.nn.Conv2d(24, 24, 5, stride=2, padding=2)
        self.conv2b = torch.nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(24, 64, 1, stride=1, padding=0)
        self.conv4 = torch.nn.Conv2d(64, 24, 3, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2a = self.conv2a(v1)
        v2b = self.conv2b(v1)
        v3 = torch.relu(v2a+v2b)
        v4 = self.conv3(v3)
        v5 = torch.relu(v4)
        v6 = self.conv4(v5)
        v7 = torch.relu(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 1, 10, 10)
