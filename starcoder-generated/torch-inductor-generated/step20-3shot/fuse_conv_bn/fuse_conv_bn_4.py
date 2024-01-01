
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 3)
        self.conv2 = torch.nn.Conv2d(3, 3, 3)
        self.bn = torch.nn.BatchNorm2d(3)
    def forward(self, x1):
        y = self.bn(self.conv1(x1) + self.conv2(x1))
        return y
# Inputs to the model
x1 = torch.randn(1, 3, 6, 6)
