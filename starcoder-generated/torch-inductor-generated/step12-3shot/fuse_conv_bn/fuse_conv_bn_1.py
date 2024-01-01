
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(4, 8, 3)
        self.bn0 = torch.nn.BatchNorm2d(8)
        self.conv1 = torch.nn.Conv2d(8, 16, 3)
        self.bn1 = torch.nn.BatchNorm2d(16)
    def forward(self, x3):
        x3 = self.bn0(self.conv0(x3))
        return self.bn1(self.conv1(x3))
# Inputs to the model
x3 = torch.randn(1, 4, 4, 4)
