
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1)
        self.conv = torch.nn.Conv2d(3, 3, 1)
        torch.manual_seed(1)
        self.bn = torch.nn.BatchNorm2d(3)
    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.bn(x1)
        x2 = self.conv(x2)
        x2 = self.bn(x2)
        return x2
# Inputs to the model
x = torch.randn(1, 3, 3, 3)
