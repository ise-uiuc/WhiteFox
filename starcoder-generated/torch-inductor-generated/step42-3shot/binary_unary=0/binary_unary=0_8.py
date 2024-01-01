
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x1, x2, x3):
        v1 = self.conv1(x1)
        v2 = x1 * v1
        v3 = torch.sigmoid(v2)
        v4 = v3 * x2
        v5 = v4 + x3
        v6 = self.conv2(v5)
        v7 = v6 + v4
        v8 = torch.relu(v7)
        v9 = self.conv3(v8)
        v10 = v9 * v7
        return v10
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
x3 = torch.randn(1, 16, 64, 64)
