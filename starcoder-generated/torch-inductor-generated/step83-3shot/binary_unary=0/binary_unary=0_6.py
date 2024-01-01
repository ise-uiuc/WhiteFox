
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(16, 16, 5, stride=1, padding=2)
    def forward(self, x1, x2, x3, x4):
        v1 = self.conv1(x1)
        v2 = torch.relu(v1)
        v3 = torch.relu(v2)
        v4 = self.conv2(v3)
        v5 = torch.relu(v4)
        v6 = torch.relu(v5)
        v7 = self.conv3(v6)
        v8 = torch.relu(v7)
        v9 = torch.relu(v8)
        v10 = v9 * x2
        v11 = torch.relu(v10 + x3)
        v12 = v11 * x4
        return torch.relu(v12)
# Inputs to the model
x1 = torch.randn(1, 16, 10, 10)
x2 = torch.randn(1, 16, 10, 10)
x3 = torch.randn(1, 16, 10, 10)
x4 = torch.randn(1, 16, 10, 10)
