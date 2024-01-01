
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x1, x2, x3, x4, x5):
        a1 = self.conv1(x3)
        v1 = a1 + x1
        v2 = torch.relu(v1)
        v3 = self.conv2(v2) + x6
        v4 = torch.relu(v6)
        v5 = self.conv3(v4) + x7
        v6 = torch.relu(v5)
        v7 = self.conv3(x5) + x8
        v8 = torch.relu(v7)
        return v8
# Inputs to the model
x18 = torch.randn(1, 16, 64, 64)
x19 = torch.randn(1, 16, 64, 64)
x20 = torch.randn(1, 16, 64, 64)
x21 = torch.randn(1, 16, 64, 64)
x22 = torch.randn(1, 16, 64, 64)
x23 = torch.randn(1, 16, 64, 64)
x24 = torch.randn(1, 16, 64, 64)
