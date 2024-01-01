
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 3)
        self.conv2 = torch.nn.Conv2d(3, 3, 2)
        self.conv3 = torch.nn.Conv2d(3, 3, 1)
        self.bn = torch.nn.BatchNorm2d(3)
    def forward(self, x2, x3):
        s = self.conv1(x2)
        t = self.conv2(x3)
        u = self.conv3(s + t)
        y = self.bn(u)
        return (s, u, y)
# Inputs to the model
x2 = torch.randn(16, 3, 6, 6)
x3 = torch.randn(1, 3, 6, 6)
