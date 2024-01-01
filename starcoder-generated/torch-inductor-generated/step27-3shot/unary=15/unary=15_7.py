
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(128, 128, 2, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = F.relu(v1)
        v3 = self.conv2(v2)
        v4 = F.relu(v3)
        v5 = self.conv3(v4)
        v6 = F.relu(v5)
        v7 = self.conv4(v6)
        v8 = F.relu(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
