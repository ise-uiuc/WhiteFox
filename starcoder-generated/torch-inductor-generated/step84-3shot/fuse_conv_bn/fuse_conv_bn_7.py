
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1)
        self.bn = torch.nn.BatchNorm2d(3)
    def forward(self, x1):
        y = torch.relu(self.bn(self.conv(self.bn(x1))))
        return self.conv(y)
# Inputs to the model
x1 = torch.randn(1, 3, 3, 3)
