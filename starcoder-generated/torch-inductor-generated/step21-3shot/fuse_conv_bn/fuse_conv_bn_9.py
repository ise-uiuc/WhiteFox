
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 8, 3)
        self.bn = torch.nn.BatchNorm2d(8)
    def forward(self, x2):
        x2 = self.conv(x2)
        x2 = self.bn(x2)
        return x2
# Inputs to the model
x2 = torch.randn(2, 4, 8, 8)
