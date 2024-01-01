
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(16, 16, 1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(16, 16, 1, stride=1, padding=0)
    def forward(self, x1, x2, x3, x4, x5):
        v1 = self.conv1(x1)
        v2 = v1 + x2
        v3 = torch.relu(v2)
        v4 = self.conv2(v3)
        v5 = torch.relu(v4 + x3)
        v6 = self.conv3(v5 + x4)
        v7 = v6 + x5
        v8 = torch.relu(v7 + v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 16, 1, 64)
x2 = torch.randn(1, 16, 1, 64)
x3 = torch.randn(1, 16, 1, 64)
x4 = torch.randn(1, 16, 1, 64)
x5 = torch.randn(1, 16, 1, 64)
