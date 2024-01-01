
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(4, 4, 3, stride=2)
        self.conv2 = torch.nn.Conv2d(4, 4, 1)
        self.bn = torch.nn.BatchNorm2d(4)
    def forward(self, x1):
        t1 = self.conv1(x1)
        t2 = self.conv2(t1)
        t2 = self.bn(t2)
        return t2
# Inputs to the model
x1 = torch.randn(1, 4, 8, 8)
