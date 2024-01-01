
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(7, 7, 2)
        self.bn1 = torch.nn.BatchNorm2d(7)
    def forward(self, x1):
        x2 = self.conv1(x1)
        x3 = self.bn1(x2)
        x4 = self.bn1(x1)
        x5 = self.conv1(x4)
        return x5
# Inputs to the model
x1 = torch.randn(1, 7, 4, 4)
