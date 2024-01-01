
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_a = torch.nn.Conv2d(3, 3, 1)
        self.conv_b = torch.nn.Conv2d(3, 3, 1)
        self.bn = torch.nn.BatchNorm2d(3)
    def forward(self, x1, x2):
        x3 = self.bn(self.conv_a(x1))
        x3 = self.bn(self.conv_b(x3))
        return x3 + self.bn(self.conv_b(x2))
# Inputs to the model
x1 = torch.randn(1, 3, 4, 4)
x2 = torch.randn(1, 3, 4, 4)
