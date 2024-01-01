
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = torch.nn.AvgPool2d(2, stride=2, padding=1, ceil_mode=True)
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x1, x2, x3):
        v1 = self.pool(x1)
        v2 = self.conv1(x1)
        v3 = v1 + v2
        v4 = torch.relu(v3)
        v5 = self.conv2(v4)
        v6 = v1 + v5
        v7 = torch.relu(v6)
        v8 = self.conv3(v7)
        v9 = v1 + v8
        v10 = torch.relu(v9)
        return v10
# Inputs to the model
x1 = torch.randn(16, 16, 64, 64)
x2 = torch.randn(16, 16, 64, 64)
x3 = torch.randn(16, 16, 64, 64)
