
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 3)
        self.conv2 = torch.nn.Conv2d(3, 3, 3)
        self.conv3 = torch.nn.Conv2d(3, 3, 3)
        self.conv4 = torch.nn.Conv2d(3, 3, 3)
        self.conv5 = torch.nn.Conv2d(3, 3, 3)
        self.bn = torch.nn.BatchNorm2d(3)
    def forward(self, x1):
        s = self.conv1(x1)
        t = self.conv2(s)
        y = self.bn(t)
        w = self.conv3(y)
        v = self.conv4(w)
        z = self.bn(v)
        m1 = self.conv5(v)
        m2 = self.conv4(z)
        return y * m2, z + m1
# Inputs to the model
x = torch.randn(1, 3, 6, 6)
