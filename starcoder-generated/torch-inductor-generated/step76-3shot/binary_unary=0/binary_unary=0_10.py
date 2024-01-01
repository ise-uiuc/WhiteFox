
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x1, x2, x3, x4):
        a1 = self.conv1(x1)
        a2 = self.conv2(x2)
        v1 = a1 + a2 + x2
        v2 = torch.relu(v1)
        a3 = self.conv3(x3)
        v3 = v2 + a3
        v4 = torch.relu(v3)
        a4 = self.conv1(x4)
        v5 = a4 + a3 + x1
        v6 = self.conv2(v5)
        return v4 + v6
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
x3 = torch.randn(1, 16, 64, 64)
x4 = torch.randn(1, 16, 64, 64)
