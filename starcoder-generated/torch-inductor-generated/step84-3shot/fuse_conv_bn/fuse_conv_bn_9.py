
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 1)
        self.bn = torch.nn.BatchNorm2d(3)
        self.conv2 = torch.nn.Conv2d(3, 3, 1)
    def forward(self, x1):
        x = self.conv1(x1)
        x = self.bn(x)
        x = x * self.conv2(x / 2)
        return x
# Inputs to the model
x1 = torch.randn(1, 3, 3, 3)
