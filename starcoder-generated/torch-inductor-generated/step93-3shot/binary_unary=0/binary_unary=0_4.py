
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 16, 5, stride=1, padding=2)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x1, x2, x3, x4):
        v1 = self.conv1(x1)
        a1 = self.conv2(v1)
        a2 = self.conv3(v1)
        v2 = a1 + a2
        v3 = torch.relu(v2)
        v4 = self.conv1(v3)
        a3 = self.conv2(v4)
        a4 = self.conv3(v4)
        v5 = a3 + a4
        v6 = torch.relu(v5)
        v7 = self.conv2(v6)
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
x3 = torch.randn(1, 16, 64, 64)
