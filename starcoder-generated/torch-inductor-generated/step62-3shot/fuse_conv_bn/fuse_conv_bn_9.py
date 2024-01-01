
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3)
        self.bn1 = torch.nn.BatchNorm2d(3)
        self.bn2 = torch.nn.BatchNorm2d(3)
        self.pool = torch.nn.AvgPool2d(2)
    def forward(self, x):
        x2 = self.pool(self.conv(x))
        x2 = self.bn1(x2)
        y2 = self.bn2(x2)
        return y2
# Inputs to the model
x = torch.randn(1, 3, 4, 4)
