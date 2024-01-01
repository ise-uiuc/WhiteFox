
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(43, 99, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(99, 3, 1, stride=1, padding=1)
        self.bn = torch.nn.BatchNorm2d(3)
        self.maxpool = torch.nn.MaxPool2d(2)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.relu(v1)
        v3 = self.conv2(v2)
        v4 = torch.relu(v3)
        v5 = self.bn(v4)
        v6 = self.maxpool(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 43, 3, 5)
