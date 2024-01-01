
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 7)
        self.pad = torch.nn.ConstantPad2d((3, 3, 2, 2), 2.0)
        self.conv2 = torch.nn.Conv2d(64, 128, 6)
        self.bn = torch.nn.BatchNorm2d(3)
    def forward(self, x3):
        y = self.conv1(x3)
        z = self.pad(x3)
        w = self.conv2(z)
        v = self.pad(y)
        u = self.conv1(v)
        a = self.bn(u)
        b = torch.tanh(a)
        return (a, b, y, z, v, w)
# Inputs to the model
x3 = torch.randn(1, 3, 10, 10)
