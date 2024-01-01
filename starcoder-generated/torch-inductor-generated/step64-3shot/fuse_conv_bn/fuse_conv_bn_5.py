
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 2, 2, groups=2)
        # For example:
        # self.bn2 = torch.nn.BatchNorm2d(2)
        self.conv2 = torch.nn.Conv2d(2, 4, 2, groups=2)
        self.bn1 = torch.nn.BatchNorm2d(4)
        self.conv3 = torch.nn.Conv2d(4, 1, 2)
    def forward(self, x1):
        x2 = self.conv1(x1)
        x2 = self.conv2(x2)
        x2 = self.bn1(x2)
        x2 = self.conv3(x2)
        x2 = self.bn1(x2)
        return x2
# Inputs to the model
x1 = torch.randn(1, 3, 3, 3)
