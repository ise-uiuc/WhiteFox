
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 7, stride=5, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 64, 3, stride=1)
        self.conv4 = torch.nn.Conv2d(64, 64, 3, stride=1)
        self.conv5 = torch.nn.Conv2d(64, 64, 3, stride=1)
        self.conv6 = torch.nn.Conv2d(64, 1, 3, stride=3, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = F.relu(v1 - 0.54)
        v3 = self.conv2(v2)
        v4 = F.relu(v3 + 0.3)
        v5 = self.conv3(v4)
        v6 = F.sigmoid(v5 - 3)
        v7 = self.conv4(v6)
        v8 = F.sigmoid(v7 + 0.53)
        v9 = self.conv5(v8)
        v10 = F.sigmoid(v9 - 2)
        v11 = self.conv6(v10)
        v12 = F.sigmoid(v11 - 1)
        v13 = F.sigmoid(v12)
        return v13.flatten(1)
# Inputs to the model
x1 = torch.randn(1, 1, 256, 256)
