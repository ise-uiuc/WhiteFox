
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3, groups=8)
        self.conv2 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1, groups=8)
        self.conv3 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1, groups=8)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(x1)
        v3 = self.conv3(x1)
        v4 = v2 + v3
        v5 = v1 + v4
        v6 = torch.relu(v5)
        v7 = self.conv1(x1)
        v8 = v7 + v6
        v9 = torch.relu(v8)
        return v9
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
