
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 128, 1, stride=1, padding=0, groups=1)
        self.bn1 = torch.nn.BatchNorm2d(128)
        self.conv2 = torch.nn.Conv2d(128, 256, 1, stride=1, padding=0, groups=1)
        self.bn2 = torch.nn.BatchNorm2d(256)
    def forward(self, x1):
        x2 = self.conv1(x1)
        x3 = self.bn1(x2)
        x4 = self.conv2(x3)
        x5 = self.bn2(x4)
        return x5
# Inputs to the model
x1 = torch.randn(1, 3, 224, 64)
