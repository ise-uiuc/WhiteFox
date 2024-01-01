
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 3)
        self.conv2 = torch.nn.Conv2d(3, 3, 3)
        self.bn = torch.nn.BatchNorm2d(3)
    def forward(self, x1, x2):
        s = self.conv1(x1)
        t = self.conv2(s)
        y = self.bn(t)
        f = s + t + y
        return f
