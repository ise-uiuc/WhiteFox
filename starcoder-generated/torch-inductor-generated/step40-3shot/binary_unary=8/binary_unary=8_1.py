
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(12, 32, 1, stride=1)
        self.conv2 = torch.nn.Conv2d(32, 32, 3, stride=3)
        self.conv3 = torch.nn.Conv2d(32, 4, 1, stride=1)
        self.conv4 = torch.nn.Conv2d(4, 4, 3, stride=3)
        self.conv5 = torch.nn.Conv2d(4, 4, 1, stride=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv1(x1)
        v3 = self.conv2(v1 + v2) + self.conv2(v1 + v2)
        v4 = self.conv3(v3) + self.conv3(v3)
        v5 = self.conv4(v4)
        v6 = self.conv5(v5)
        v7 = torch.relu(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 12, 64, 64)
