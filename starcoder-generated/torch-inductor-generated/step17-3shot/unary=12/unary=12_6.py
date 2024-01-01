
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 32, 3, 2, 1)
        self.relu = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool2d(3, 2, 0)
        self.bn = torch.nn.BatchNorm2d(32)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.relu(v1)
        v3 = self.maxpool(v2)
        v4 = self.bn(v3)
        v5 = self.relu(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
