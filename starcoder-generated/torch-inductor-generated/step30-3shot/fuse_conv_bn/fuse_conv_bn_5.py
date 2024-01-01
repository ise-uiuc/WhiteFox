
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(20, 10, 5)
        self.conv2 = torch.nn.Conv2d(10, 5, 3)
        self.conv3 = torch.nn.Conv2d(5, 30, 1)
        self.bn1 = torch.nn.BatchNorm2d(20)
        self.bn2 = torch.nn.BatchNorm2d(10)
        self.bn3 = torch.nn.BatchNorm2d(5)
    def forward(self, x1):
        y1 = self.conv1(x1)
        y2 = self.conv2(y1)
        y3 = self.conv3(y2)
        y4 = self.bn1(x1)
        y5 = self.bn2(y4)
        y6 = self.bn3(y5)
        y7 = y3 + y6
        return y7
# Inputs to the model
x2 = torch.randn(1, 20, 28, 28)
