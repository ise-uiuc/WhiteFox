
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=3, padding=3, dilation=2)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=3, padding=3, dilation=2)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=3, padding=3, dilation=2)
    def forward(self, x1, x2, x3, x4, x5):
        v1 = self.conv1(x1)
        v2 = self.conv2(x2)
        v3 = v1 + v2
        v4 = torch.relu(v3)
        v5 = self.conv3(v4)
        v6 = v5 + x3
        v7 = torch.relu(v6)
        v8 = v7 + x4
        v9 = torch.relu(v8)
        return v9
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
x3 = torch.randn(1, 16, 64, 64)
x4 = torch.randn(1, 16, 64, 64)
x5 = torch.randn(1, 16, 64, 64)
