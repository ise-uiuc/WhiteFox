
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x1, x2, x3):
        v0 = x1.mean((2,3))
        v1 = self.conv1(v0)
        v2 = v1.min((2,3))
        v3 = self.conv2(v2)
        v4 = v3 * x2
        v5 = torch.relu(v4)
        v6 = v2 + v5
        v7 = torch.relu(v6)
        v8 = v7.amin((2,3))
        v9 = self.conv3(v8)
        v10 = v9 + x3
        v11 = torch.relu(v10)
        v12 = v10 - v11
        return v12
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
x3 = torch.randn(1, 16, 64, 64)
