
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(4, 4, 3, stride=2)
        self.conv2 = torch.nn.Conv2d(4, 4, 1)
        self.bn = torch.nn.BatchNorm2d(4)
    def forward(self, x1):
        y1 = self.conv1(x1)
        y2 = self.conv2(y1)
        y2 = self.bn(y2)
        return y2
# Inputs to the model
x1 = torch.randn(1, 4, 7, 7)
