
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(8, 8, 2, stride=1)
        self.conv2 = torch.nn.Conv2d(8, 8, 4, stride=1)
        self.conv3 = torch.nn.Conv2d(8, 8, 3, stride=1)
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.bn2 = torch.nn.BatchNorm2d(8)
        self.relu = torch.nn.ReLU(inplace=True)
    def forward(self, x1):
        x2 = self.relu(self.bn1(self.conv1(x1)))
        x3 = self.relu(self.bn1(self.conv2(x2)))
        x4 = self.relu(self.bn1(self.conv3(x3)))
        return x4
# Inputs to the model
x1 = torch.randn(1, 8, 12, 12)
