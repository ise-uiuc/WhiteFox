
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 2)
        self.conv_bn1 = torch.nn.Conv2d(3, 3, 2)
        self.bn1 = torch.nn.BatchNorm2d(3)
    def forward(self, x1):
        x2 = self.conv1(x1)
        x4 = self.bn1(x2)
        x3 = self.conv_bn1(x4)
        return x3
# Inputs to the model
x1 = torch.randn(1, 3, 4, 4)
