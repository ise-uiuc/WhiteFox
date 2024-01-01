
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 2, 3)
        self.bn = torch.nn.BatchNorm2d(2)
        self.conv2 = torch.nn.Conv2d(2, 1, 3)
    def forward(self, x2):
        y1 = self.conv1(x2)
        y2 = self.bn(y1)
        y3 = self.conv2(y2)
        return y3
# Inputs to the model
x2 = torch.randn(1, 3, 8, 8)
