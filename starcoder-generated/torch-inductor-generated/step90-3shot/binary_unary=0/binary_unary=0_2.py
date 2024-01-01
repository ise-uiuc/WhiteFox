
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 1, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(1, 16, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(16, 1, 3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(1, 16, 3, stride=1, padding=1)
    def forward(self, x1, x2, x3, x4):
        v1 = self.conv1(x1)
        v2 = torch.relu(v1)
        v3 = self.conv2(v1)
        v4 = torch.relu(v3)
        v5 = v3 + self.conv3(v2)
        v6 = torch.relu(v5)
        v7 = self.conv4(x1)
        v8 = torch.relu(v3)
        v9 = x2
        v10 = self.conv4(v7)
        v11 = torch.relu(v10)
        return v11
# Inputs to the model
x1 = torch.randn(1, 16, 8, 8)
x2 = torch.randn(1, 16, 8, 8)
x3 = torch.randn(1, 16, 8, 8)
x4 = torch.randn(1, 16, 8, 8)
