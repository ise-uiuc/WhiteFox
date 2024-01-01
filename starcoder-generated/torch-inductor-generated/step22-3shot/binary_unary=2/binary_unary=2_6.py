
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 64, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(64, 16, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(16, 8, 5, stride=1, padding=2)
        self.conv4 = torch.nn.Conv2d(8, 1, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 - 2
        v3 = F.relu(v2)
        v4 = self.conv2(v3)
        v5 = v4 - 6.0
        v6 = F.relu(v5)
        v7 = self.conv3(v6)
        v8 = v7 - 0.5
        v9 = F.relu(v8)
        v10 = v9 - 6.0
        v11 = F.sigmoid(v10)
        v12 = self.conv4(v11)
        v13 = v12 - 0.5
        return v13
# Inputs to the model
x1 = torch.randn(1, 1, 28, 28)
