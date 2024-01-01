
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(2, 2, 3)
        self.bn = torch.nn.BatchNorm2d(2)
    def forward(self, x):
        x1 = self.conv1(x)
        y2 = self.bn(x1)
        return y2
# Inputs to the model
x1 = torch.randn(2, 2, 4, 4)
