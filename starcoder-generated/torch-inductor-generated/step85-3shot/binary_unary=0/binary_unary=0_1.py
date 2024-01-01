
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 32, 5, stride=1, padding=2, groups=2)
        self.conv2 = torch.nn.Conv2d(96, 48, 5, stride=2, padding=2, groups=36)
        self.conv3 = torch.nn.Conv2d(272, 64, 3, stride=2, padding=1, groups=72)
    def forward(self, x1, x2, x3, x4):
        v1 = self.conv1(x1)
        b1 = self.conv2(x2)
        v2 = v1 + x3
        v3 = b1 + x2
        v4 = torch.relu(v2)
        v5 = self.conv3(v4)
        v6 = v5 + x4
        v7 = torch.relu(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 32, 32)
x3 = torch.randn(1, 32, 64, 64)
x4 = torch.randn(1, 48, 32, 32)
