
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__(x1, x2):
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(8)
    def forward(self):
        v1 = self.conv1(x1)
        v2 = self.conv2(x2)
        bn1 = self.bn1(v2)
        return bn1
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
x2 = torch.randn(1, 3, 32, 32)
