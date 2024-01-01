
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x1, x2, x3, x4):
        a1 = self.conv1(x1)
        v1 = a1 + x1
        a2 = self.conv2(v1) + v1
        a3 = self.conv3(x4)
        v2 = a2 + x2
        v3 = torch.relu(v2)
        v4 = a3 + x2
        v5 = torch.relu(v4)
        v6 = self.conv3(v5)
        v7 = v6 + x3
        v8 = torch.relu(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
x3 = torch.randn(1, 16, 64, 64)
x4 = torch.randn(1, 16, 64, 64)
