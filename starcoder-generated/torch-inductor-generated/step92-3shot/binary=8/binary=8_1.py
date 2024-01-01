
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=2)
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.conv2 = torch.nn.Conv2d(8, 8, 1)
        self.bn2 = torch.nn.BatchNorm2d(8)
    def forward(self, x):
        v = self.conv1(x)
        v = self.bn1(v)
        v = self.conv2(v)
        v = self.bn2(v)
        return v
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
