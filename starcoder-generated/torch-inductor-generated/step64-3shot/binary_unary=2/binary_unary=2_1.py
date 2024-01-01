
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, 1, stride=1)
        self.conv3 = torch.nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = torch.nn.Conv2d(32, 9, 1, stride=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 - 0
        v3 = F.relu(v2)
        v4 = self.conv2(v3)
        v5 = v4 - 1
        v6 = F.relu(v5)
        v7 = self.conv4(v6)
        v8 = v7 - 0
        v9 = F.relu(v8)
        return v9
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
