
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3, groups=2)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3, groups=2)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3, groups=2)
        self.conv4 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3, groups=2)
    def forward(self, x1, x2, x3, x4):
        v1 = self.conv1(x1)
        v2 = v1 + x1
        v3 = torch.tanh(v2)
        v4 = self.conv2(v3)
        v5 = v4 + x2
        v6 = torch.sin(v5)
        v7 = self.conv3(v6)
        v8 = v7 + x3
        v9 = torch.relu(v8)
        return v9
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
x3 = torch.randn(1, 16, 64, 64)
x4 = torch.randn(1, 16, 64, 64)
