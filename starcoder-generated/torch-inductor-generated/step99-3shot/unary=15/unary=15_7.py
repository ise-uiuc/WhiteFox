
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(784, 256, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(256, 128, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(128, 10, 3, stride=1, padding=1)
        self.pool = torch.nn.AvgPool2d(2, stride=2)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.relu(v1)
        v3 = self.pool(v2)
        v4 = self.conv2(v3)
        v5 = torch.relu(v4)
        v6 = self.pool(v5)
        v7 = self.conv3(v6)
        v8 = torch.relu(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 1, 28, 28)
