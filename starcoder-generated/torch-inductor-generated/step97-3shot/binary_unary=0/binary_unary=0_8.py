
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 32, 7, stride=2, padding=3)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv4 = torch.nn.Conv2d(128, 32, 3, stride=2, padding=1)
    def forward(self, x, y):
        v1 = self.conv1(x)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = self.conv4(v3)
        v5 = v4 * x - y
        v6 = torch.relu(v5)
        v7 = v6 * x + y
        v8 = torch.relu(v7)
        v9 = v8 * x
        return v9
# Inputs to the model
x = torch.randn(1, 16, 64, 64)
y = torch.randn(1, 16, 64, 64)
