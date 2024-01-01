
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 8, 3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 9, 3, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(9, 5, 3, stride=2, padding=1)
        self.conv4 = torch.nn.Conv2d(5, 1, 3, stride=2, padding=1)
        self.conv5 = torch.nn.Conv2d(1, 8, 3, stride=2, padding=1)
        self.conv6 = torch.nn.Conv2d(8, 1, 3, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 * 0.1 - 3
        v3 = F.relu(v2)
        v4 = self.conv2(v3)
        v5 = v4 * 0.1 - 3
        v6 = F.relu(v5)
        v7 = self.conv3(v6)
        v8 = v7 * 0.1 - 3
        v9 = F.relu(v8)
        v10 = self.conv4(v9)
        v11 = v10 * 0.1 - 3
        v12 = F.relu(v11)
        v13 = self.conv5(v12)
        v14 = v13 * 0.1 - 3
        v15 = F.relu(v14)
        return self.conv6(v15)
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
