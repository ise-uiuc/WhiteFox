
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 32, 16, padding=8, bias=False)
        self.bn = torch.nn.BatchNorm2d(32)
    def forward(self, x):
        x = self.conv(x)
        y = self.bn(x)
        return y
# Inputs to the model
x = torch.randn(1, 3, 224, 224)
