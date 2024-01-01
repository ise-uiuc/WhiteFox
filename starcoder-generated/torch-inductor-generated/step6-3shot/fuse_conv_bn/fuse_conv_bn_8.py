
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3, padding=1)
        self.bn = torch.nn.BatchNorm2d(3)
    def forward(self, x):
        return self.bn(self.conv(x))
# Inputs to the model
x = torch.randn(1, 3, 5, 5)
