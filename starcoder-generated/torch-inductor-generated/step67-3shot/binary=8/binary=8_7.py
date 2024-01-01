
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 7, stride=2, padding=3)
        self.conv2 = torch.nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv5 = torch.nn.Conv2d(32, 1, 3, stride=1, padding=1)
        self.conv6 = torch.nn.Conv2d(32, 1, 3, stride=1, padding=1)
        self.relu = torch.nn.ReLU(inplace=True)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = self.relu(v1)
        v3 = self.conv2(v2)
        v4 = self.relu(v3)
        v5 = self.conv3(v4)
        v6 = self.relu(v5)
        v7 = self.conv4(v6)
        v8 = self.relu(v7)
        v9 = self.conv5(v8)
        v10 = self.relu(v7)
        v11 = self.conv6(v10)
        v12 = self.relu(v11)
        v13 = v9 + v12
        return v13
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
x2 = torch.randn(1, 3, 224, 224)
