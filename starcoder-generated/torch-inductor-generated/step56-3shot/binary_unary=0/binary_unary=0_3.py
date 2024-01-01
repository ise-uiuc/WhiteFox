
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 32, 7, stride=2, padding=3)
        self.conv3 = torch.nn.Conv2d(32, 64, 11, stride=2, padding=5)
        self.conv4 = torch.nn.Conv2d(64, 16, 13, stride=1, padding=15)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = torch.relu(v1)
        v3 = self.conv2(v2)
        v4 = torch.relu(v3)
        v5 = self.conv3(v4)
        v6 = torch.relu(v5)
        v7 = self.conv4(v6)
        return v7
# Inputs to the model
x = torch.randn(1, 1, 224, 224)
