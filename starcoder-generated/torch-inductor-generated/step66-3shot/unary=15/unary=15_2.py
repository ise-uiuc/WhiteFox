
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 1, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.pool1 = torch.nn.MaxPool2d(2, stride=2)
        self.conv1_1 = torch.nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.bn1_2 = torch.nn.BatchNorm2d(32)
        self.conv1_2 = torch.nn.Conv2d(32, 32, 1, stride=1, padding=1)
        self.conv2_1 = torch.nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.bn2_2 = torch.nn.BatchNorm2d(32)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.bn1(v1)
        v3 = torch.relu(v2)
        v4 = self.pool1(v3)
        v5 = self.conv1_1(v4)
        v6 = self.bn1_2(v5)
        v7 = torch.relu(v6)
        v8 = self.conv1_2(v7)
        v9 = self.conv2_1(v8)
        v10 = self.bn2_2(v9)
        v11 = torch.relu(v10)
        return v11
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
