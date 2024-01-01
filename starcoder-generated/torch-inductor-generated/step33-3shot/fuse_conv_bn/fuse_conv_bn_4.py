
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3)
        self.conv3 = torch.nn.Conv2d(3, 3, 1)
        self.bn = torch.nn.BatchNorm2d(3)
        self.relu6 = torch.nn.ReLU6()
    def forward(self, x3):
        r = self.conv(x3)
        x = self.conv3(r)
        v = self.relu6(x)
        e = self.bn(v).sum()
        return (e)
# Inputs to the model
x3 = torch.randn(1, 3, 5, 5)
