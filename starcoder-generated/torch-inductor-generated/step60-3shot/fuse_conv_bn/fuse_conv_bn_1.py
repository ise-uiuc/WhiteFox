
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 2, 3)
        self.pad = torch.nn.ReflectionPad2d(1)
        self.bn = torch.nn.BatchNorm2d(2)
    def forward(self, x):
        x = self.conv(x)
        x = self.pad(x)
        x = self.bn(x)
        return x
# Inputs to the model
x = torch.randn(1, 3, 3, 3)
