
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(4, 4, 1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(4)
        self.conv1d = torch.nn.Conv2d(4, 4, 1, bias=False)
        self.bn1d = torch.nn.BatchNorm2d(4)
        self.conv2 = torch.nn.Conv2d(4, 4, 3, padding=1, bias=False)
        self.bn = torch.nn.BatchNorm2d(4)
        self.conv3d = torch.nn.Conv2d(4, 4, 3, padding=1, bias=False)
        self.bn3d = torch.nn.BatchNorm2d(4)
    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = x + self.bn1d(self.conv1d(x))
        x = self.bn(self.conv2(x))
        x = self.bn3d(self.conv3d(x))
        return x
# Inputs to the model
x = torch.randn(1, 4, 6, 6)
