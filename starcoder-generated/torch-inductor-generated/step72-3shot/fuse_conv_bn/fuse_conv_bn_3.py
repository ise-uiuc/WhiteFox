
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(4)
        self.conv = torch.nn.Conv2d(4, 2, 3, stride=1)
        torch.manual_seed(4)
        self.bn = torch.nn.BatchNorm2d(2)
    def forward(self, x3):
        x3 = self.conv(x3)
        x3 = self.conv(x3)
        x3 = self.conv(x3)
        x3 = self.bn(x3)
        x3 = self.conv(x3)
        return x3
# Inputs to the model
x3 = torch.randn(1, 4, 16, 16)
