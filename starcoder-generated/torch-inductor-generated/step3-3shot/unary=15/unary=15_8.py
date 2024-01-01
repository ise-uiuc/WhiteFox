
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 3, stride=2, padding=1)
        self.bn = torch.nn.BatchNorm2d(32)
        self.conv2 = torch.nn.Conv2d(32, 32, 1, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d((5, 5), stride=(5, 5), padding=2)
        self.conv3 = torch.nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(64)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.bn(v1)
        v3 = torch.relu(v2)
        v4 = self.conv2(v3)
        v5 = self.pool(v4)
        v6 = self.conv3(v5)
        v7 = self.bn2(v6)
        v8 = torch.relu(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
