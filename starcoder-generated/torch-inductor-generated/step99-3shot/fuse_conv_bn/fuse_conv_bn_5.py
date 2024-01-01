
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3)
        self.relu = torch.nn.ReLU()
        self.avgpool = torch.nn.AvgPool2d(3, 1, 1)
        self.sigmoid = torch.nn.Sigmoid()
        self.bn1 = torch.nn.BatchNorm2d(3)
        self.bn2 = torch.nn.BatchNorm2d(3)
    def forward(self, x1):
        x1 = self.conv(x1)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x1 = self.avgpool(x1)
        x1 = self.bn2(x1)
        x1 = self.sigmoid(x1)
        return x1 