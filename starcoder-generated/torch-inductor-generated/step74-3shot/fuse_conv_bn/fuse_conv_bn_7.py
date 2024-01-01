
class Model(torch.nn.Module):
    def __init__(self, a):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, 7, stride=2, padding=(3, 3))
        self.bn1 = torch.nn.BatchNorm2d(1)
        self.conv2 = torch.nn.Conv2d(1, 2, 7, stride=2, padding=(3, 3))
        self.bn2 = torch.nn.BatchNorm2d(2)
        self.relu = torch.nn.ReLU()
        self.conv3 = torch.nn.Conv2d(2, 2, 3, stride=2, padding=(1, 1))
        self.bn3 = torch.nn.BatchNorm2d(2)
        self.conv4 = torch.nn.Conv2d(2, 4, 3, stride=2, padding=(1, 1))
        self.bn4 = torch.nn.BatchNorm2d(4)
        self.conv5 = torch.nn.Conv2d(4, 4, 3, stride=2, padding=(1, 1))
        self.bn5 = torch.nn.BatchNorm2d(4)
    def forward(self, x3):
        v1 = self.conv1(x3)
        v2 = self.bn1(v1)
        v3 = self.relu(v2)
        v4 = self.conv2(v3)
        v5 = self.bn2(v4)
        v6 = self.relu(v5)
        v7 = self.conv3(v6)
        v8 = self.bn3(v7)
        v9 = self.relu(v8)
        v10 = self.conv4(v9)
        v10 = self.bn4(v10)
        v12 = self.conv5(v10)
        v12 = self.bn5(v12)
        return v7, v12
# Inputs to the model
x3 = torch.randn(3, 1, 10, 20)
