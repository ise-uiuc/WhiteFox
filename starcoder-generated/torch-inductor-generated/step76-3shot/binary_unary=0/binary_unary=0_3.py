
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv4 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv5 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x1, x2, x3, x4):
        a1 = self.conv1(x2)
        v1 = a1 + x2 + x3
        v2 = torch.relu(v1)
        v3 = self.conv2(v2) + x1
        v4 = torch.relu(v3)
        v5 = self.conv3(v4) + x4
        v6 = torch.relu(v5)
        v7 = self.conv4(x1) + x3
        v8 = torch.relu(v7)
        v9 = self.conv5(v8) + x2
        v10 = torch.relu(v9)
        return v10
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
x3 = torch.randn(1, 16, 64, 64)
x4 = torch.randn(1, 16, 64, 64)
