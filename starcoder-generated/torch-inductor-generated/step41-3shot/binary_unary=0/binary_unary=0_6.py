
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x1, x2, x3):
        v1 = self.conv1(x1)
        v2 = self.conv2(x1)
        v3 = v1 + x2
        v4 = torch.relu(v3)
        v5 = v3 + v4
        v6 = torch.relu(v5)
        v7 = v6 + x3
        v8 = torch.relu(v7)
        v9 = self.conv3(v8)
        v10 = v9 + x1
        v11 = torch.relu(v10)
        return v11
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
x3 = torch.randn(1, 16, 64, 64)
