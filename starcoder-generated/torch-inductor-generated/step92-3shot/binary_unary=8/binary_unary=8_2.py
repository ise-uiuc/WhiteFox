
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 8, 3, stride=2, padding=0)
        self.conv2 = torch.nn.Conv2d(1, 16, 3, stride=2, padding=0)
        self.conv3 = torch.nn.Conv2d(1, 16, 3, stride=2, padding=0)
        self.conv4 = torch.nn.Conv2d(1, 24, 3, stride=2, padding=0)
        self.conv5 = torch.nn.Conv2d(1, 32, 3, stride=2, padding=0)
        self.conv6 = torch.nn.Conv2d(1, 32, 3, stride=2, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(x1)
        v3 = self.conv3(x1)
        v4 = self.conv4(x1)
        v5 = self.conv5(x1)
        v6 = self.conv6(x1)
        v7 = torch.relu(v1)
        v8 = torch.relu(v2)
        v9 = torch.relu(v3)
        v10 = torch.relu(v4)
        v11 = torch.relu(v5)
        v12 = torch.relu(v6)
        v13 = v1 + v2 + v3 + v4 + v5 + v6
        v14 = torch.relu(v13)
        return v14
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
