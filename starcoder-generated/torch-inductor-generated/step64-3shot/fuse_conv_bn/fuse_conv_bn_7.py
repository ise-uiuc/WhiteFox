
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 4, 1)
        self.conv2 = torch.nn.Conv2d(5, 6, 1)
        self.bn = torch.nn.BatchNorm2d(6)
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x = self.bn(torch.cat([x1, x2], 1))
        return x
# Inputs to the model
x = torch.randn(1, 3, 3, 3)
