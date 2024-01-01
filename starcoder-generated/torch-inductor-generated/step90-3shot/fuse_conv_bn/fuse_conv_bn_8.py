
class Model(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 5)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU(True)
        self.max2 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 10, 3)
    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.max2(x)
        return self.conv2(x)
# Inputs to the model
x = torch.randn(1, 3, 4)
