
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 3, stride=1, padding=1)
        self.pool1 = torch.nn.MaxPool2d(2, stride=2)
        self.conv2 = torch.nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.pool2 = torch.nn.MaxPool2d(2, stride=2)
        self.conv3 = torch.nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(64, 1, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 - 1.01
        v3 = F.relu(v2)
        v4 = self.pool1(v3)
        v5 = self.conv2(v4)
        v6 = v5 - 2.02
        v7 = F.relu(v6)
        v8 = self.pool2(v7)
        v9 = self.conv3(v8)
        v10 = v9 - 3.03
        v11 = F.relu(v10)
        v12 = self.conv4(v11)
        v13 = v12 - 4.04
        v14 = F.relu(v13)
        return v14
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
