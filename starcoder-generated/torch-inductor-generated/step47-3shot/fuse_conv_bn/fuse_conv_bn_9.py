
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 2)
        self.bn = torch.nn.BatchNorm2d(1)
    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.bn(x)
        return x
# Inputs to the model
x = torch.randn(1, 3, 16, 16)
