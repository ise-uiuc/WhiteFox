
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 4, 3)
        self.bn = torch.nn.BatchNorm2d(4)
        self.flatten= torch.nn.Flatten()
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.flatten(x)
        x = self.conv(x)
        return x
# Inputs to the model
x = torch.randn(1, 4, 3, 3)
