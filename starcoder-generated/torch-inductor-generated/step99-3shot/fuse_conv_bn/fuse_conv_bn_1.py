
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 3, 3)
        self.conv_bn = torch.nn.Sequential(self.conv, torch.nn.BatchNorm2d(3))
    def forward(self, x1):
        x2 = self.conv_bn(x1)
        return x2
# Inputs to the model
x1 = torch.randn(1, 2, 4, 4)
