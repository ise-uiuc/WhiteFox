
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(4, 32, 3, stride=2, padding=1, groups=4)
        self.conv2 = torch.nn.Conv2d(32, 16, 1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(16, 32, 3, stride=2, padding=1, groups=4)
        self.conv4 = torch.nn.Conv2d(32, 32, 1, stride=1, padding=0)
        self.conv5 = torch.nn.Conv2d(32, 1, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.relu(v1)
        v3 = self.conv2(v2)
        v4 = torch.relu(v3)
        v5 = self.conv3(v4)
        v6 = torch.relu(v5)
        v7 = self.conv4(v6)
        v8 = torch.relu(v7)
        v9 = self.conv5(v8)
        v10 = torch.relu(v9)
        return v10
# Inputs to the model
x1 = torch.randn(1, 4, 32, 32)
