
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, 1,)
        self.conv2 = torch.nn.Conv2d(1, 1, 2)
        self.conv3 = torch.nn.Conv2d(1, 1, 3)
        self.bn = torch.nn.BatchNorm2d(1)
    def forward(self, x):
        x = self.conv1(x)
        y = self.conv2(x)
        z = self.conv3(x)
        y = self.bn(y)
        o = y + z
        return o
# Inputs to the model
X = torch.randn(1, 1, 1, 1)
