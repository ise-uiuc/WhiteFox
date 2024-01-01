
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=0)
        self.conv4 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x5, x6):
        v1 = self.conv1(x5)
        v2 = self.conv2(x5)
        v3 = v1 + x6
        v4 = v2 + x6
        v5 = torch.relu(v3)
        v6 = torch.relu(v4)
        v7 = v5
        v8 = self.conv3(v6)
        v9 = v8 + x6
        v10 = torch.relu(v9)
        v11 = self.conv4(x5)
        v12 = self.conv1(x6)
        v13 = v11 + v12
        v14 = self.conv3(v13)
        v15 = v14 * x5
        return v15
# Inputs to the model
x4 = torch.randn(1, 16, 64, 64)
x5 = torch.randn(1, 16, 64, 64)
