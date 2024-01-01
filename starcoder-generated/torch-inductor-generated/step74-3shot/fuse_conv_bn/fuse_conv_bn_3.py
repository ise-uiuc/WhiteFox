
class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 4, 1, stride=1)
        self.conv2 = torch.nn.Conv2d(4, 5, 1, stride=1)
        self.bn = torch.nn.BatchNorm2d(5)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.conv2(x)
        return x
# Inputs to the model
x = torch.randn(1, 3, 32, 32)
