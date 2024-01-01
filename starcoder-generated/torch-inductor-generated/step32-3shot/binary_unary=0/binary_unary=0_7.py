
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1)
    def forward(self, x1, x2, x3):
        v1 = x1 + x1
        v2 = v1 + x2
        v3 = self.conv1(v2)
        v4 = torch.relu(v3)
        v5 = v4 - x3
        v6 = torch.relu(v5)
        v7 = v6 - v6
        v8 = self.conv2(v7)
        v9 = torch.relu(v8)
        v10 = self.conv3(v9)
        return v10
# Inputs to the model
x1 = torch.randn(2, 16, 64, 64)
x2 = torch.randn(2, 16, 64, 64)
x3 = torch.randn(2, 16, 64, 64)
