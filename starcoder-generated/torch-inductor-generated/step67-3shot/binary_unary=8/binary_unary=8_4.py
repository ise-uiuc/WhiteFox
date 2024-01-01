
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, 3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(1, 16, 3, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(1, 8, 3, stride=2, padding=1)
        self.conv4 = torch.nn.Conv2d(1, 8, 3, stride=2, padding=1)
        self.conv5 = torch.nn.Conv2d(1, 8, 3, stride=2, padding=1)
        self.conv6 = torch.nn.Conv2d(1, 4, 3, stride=2, padding=1)
        self.conv7 = torch.nn.Conv2d(1, 4, 3, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(x1)
        v3 = self.conv3(x1)
        v4 = self.conv4(x1)
        v5 = self.conv5(x1)
        v6 = self.conv6(x1)
        v7 = self.conv7(x1)
        v8 = v1 + v2 + v3 + v4
        v9 = torch.relu(v8)
        v10 = v3 + v4 + v5 + v6
        v11 = torch.relu(v10)
        v12 = v1 + v2 + v3 + v4 + v5 + v6 + v7
        v13 = torch.relu(v12)
        return v9+v11+v13
# Inputs to the model
x1 = torch.randn(1, 1, 7, 7)
