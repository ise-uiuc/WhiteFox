
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3, groups=4)
        self.depthwise1 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1, groups=16)
        self.depthwise2 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1, groups=16)
        self.conv2 = torch.nn.Conv2d(16, 16, 1, stride=1, padding=0)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = self.depthwise1(x1)
        v3 = self.depthwise2(x1)
        v4 = v1 + x2
        v5 = torch.relu(v4)
        v6 = v5 + v2
        v7 = torch.relu(v6)
        v8 = self.conv2(v7)
        v9 = v8 + v3
        v10 = torch.relu(v9)
        return v10
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
