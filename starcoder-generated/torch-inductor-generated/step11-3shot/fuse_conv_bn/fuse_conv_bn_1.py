
class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 3, 3)
        self.conv2 = torch.nn.Conv2d(3, 3, 3)
        self.bn = torch.nn.BatchNorm2d(3)
    def forward(self, x1):
        x1 = self.conv1(x1)
        y = self.bn(x1)
        y = self.bn(y)
        if y.size(0) == 1:
            y = self.conv2(y)
        else:
            y = self.conv1(y)
        return y
# Inputs to the model
x1 = torch.randn(2, 1, 4, 4)
