
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 3)
        self.conv2 = torch.nn.Conv2d(3, 3, 3)
        self.conv3 = torch.nn.Conv2d(3, 3, 3)
        self.bn = torch.nn.BatchNorm2d(3)
    def forward(self, x):
        v = self.conv1(x)
        v = self.conv2(v)
        v = self.conv3(v)
        v = self.bn(v)
        return v
# Inputs to the model
x = torch.randn(1, 3, 5, 5)
