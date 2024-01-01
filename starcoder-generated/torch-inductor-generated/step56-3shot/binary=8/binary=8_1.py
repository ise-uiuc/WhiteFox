
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.conv1 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=0)
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.bn2 = torch.nn.BatchNorm2d(8)
        self.conv4 = torch.nn.Conv2d(8, 1, 1, stride=1, padding=0)
    def forward(self, x1, x2):
        v1 = self.relu(self.conv1(x1))
        v2 = self.relu(self.conv2(x2))
        v3 = self.relu(x1)
        v4 = self.relu(self.conv3(v3))
        v5 = self.bn1(v1)
        v6 = self.bn2(v2)
        v7 = v5 + v6
        v8 = self.relu(v7)
        v9 = self.conv4(v8)
        return v9
# Inputs to the model
x1 = torch.randn(1, 3, 50, 50)
x2 = torch.randn(1, 3, 50, 50)
