
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3, groups=2)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3, groups=2)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3, groups=1)
        self.conv4 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3, groups=1)
    def forward(self, x1, x2, x3):
        v1 = self.conv1(x1)
        v2 = self.conv2(x1)
        v3 = self.conv3(x1)
        a1 = v3 + v1
        v4 = torch.relu(a1)
        a2 = v1 + v3
        v5 = torch.relu(a2)
        v6 = self.conv4(v4)
        a3 = v5 + v2
        v7 = torch.relu(a3)
        a4 = v7 + v4
        v8 = torch.relu(a4)
        a5 = v7 + v6
        v9 = torch.relu(a5)
        v10 = self.conv2(v7)
        a6 = v8 + v10
        v11 = torch.relu(a6)
        return v11
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
x3 = torch.randn(1, 16, 64, 64)
