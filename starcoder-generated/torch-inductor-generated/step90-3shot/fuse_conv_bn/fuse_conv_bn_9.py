
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3)
        self.bn = torch.nn.BatchNorm2d(3)
        self.relu = torch.nn.ReLU(inplace=True)
    def forward(self, x3, x4):
        y = self.relu(self.bn(self.conv(x3)))
        return self.relu(self.bn(self.conv(y)))
# Inputs to the model
x3 = torch.randn(1, 3, 3, 3)
x4 = torch.randn(1, 3, 3, 3)
