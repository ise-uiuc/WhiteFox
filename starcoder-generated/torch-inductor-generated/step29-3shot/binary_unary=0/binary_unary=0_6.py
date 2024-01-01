
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(16, 16, 1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(16, 16, 1, stride=1, padding=0)
    def forward(self, x1, x2, x3, x4, x5):
        v1 = self.conv1(x1)
        v2 = v1 + x2
        v3 = torch.relu(v2)
        v4 = self.conv2(v3)
        v5 = x3 + v4
        v6 = torch.relu(v5)
        v7 = v2 + v6
        v8 = self.conv3(v7)
        v9 = v8 + x4
        v10 = torch.relu(v9 + x5)
        return v10
# Inputs to the model
x1 = torch.randn(1, 16, 1, 64)
x2 = torch.randn(1, 16, 1, 64)
x3 = torch.randn(1, 16, 1, 64)
x4 = torch.randn(1, 16, 1, 64)
x5 = torch.randn(1, 16, 1, 64)
