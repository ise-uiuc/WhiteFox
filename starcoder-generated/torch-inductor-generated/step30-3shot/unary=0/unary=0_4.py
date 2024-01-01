
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 4, 3, stride=2)
        self.bn1 = torch.nn.BatchNorm2d(4)
        self.conv2 = torch.nn.Conv2d(4, 16, 3, stride=2, groups=2)
        self.bn2 = torch.nn.BatchNorm2d(16)
        self.relu = torch.nn.ReLU()
        self.flatten = torch.nn.Flatten()
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.bn1(v1)
        v3 = self.relu(v2)
        v4 = self.conv2(v3)
        v5 = self.bn2(v4)
        v6 = self.relu(v5)
        v7 = self.flatten(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 512, 512)
