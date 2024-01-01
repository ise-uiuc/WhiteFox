
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x1, x2, x3, x4):
        a1 = self.conv1(x1)
        v1 = self.conv2(x2)
        v2 = torch.relu(a1)
        v3 = v2 + x3
        v4 = torch.relu(v3)
        v5 = v1 + x4
        v6 = torch.relu(v5)
        v7 = v4 + v6
        v8 = torch.relu(v7)
        v9 = v8 + x1
        v10 = torch.relu(v9)
        return v10
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
x3 = torch.randn(1, 16, 64, 64)
x4 = torch.randn(1, 16, 64, 64)
